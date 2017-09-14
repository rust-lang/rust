// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

use errors::DiagnosticBuilder;
use hir::HirId;
use ich::StableHashingContext;
use lint::builtin;
use lint::context::CheckLintNameResult;
use lint::{self, Lint, LintId, Level, LintSource};
use rustc_data_structures::stable_hasher::{HashStable, ToStableHashKey,
                                           StableHasher, StableHasherResult};
use session::Session;
use syntax::ast;
use syntax::attr;
use syntax::codemap::MultiSpan;
use syntax::symbol::Symbol;
use util::nodemap::FxHashMap;

pub struct LintLevelSets {
    list: Vec<LintSet>,
    lint_cap: Level,
}

enum LintSet {
    CommandLine {
        // -A,-W,-D flags, a `Symbol` for the flag itself and `Level` for which
        // flag.
        specs: FxHashMap<LintId, (Level, LintSource)>,
    },

    Node {
        specs: FxHashMap<LintId, (Level, LintSource)>,
        parent: u32,
    },
}

impl LintLevelSets {
    pub fn new(sess: &Session) -> LintLevelSets {
        let mut me = LintLevelSets {
            list: Vec::new(),
            lint_cap: Level::Forbid,
        };
        me.process_command_line(sess);
        return me
    }

    pub fn builder(sess: &Session) -> LintLevelsBuilder {
        LintLevelsBuilder::new(sess, LintLevelSets::new(sess))
    }

    fn process_command_line(&mut self, sess: &Session) {
        let store = sess.lint_store.borrow();
        let mut specs = FxHashMap();
        self.lint_cap = sess.opts.lint_cap.unwrap_or(Level::Forbid);

        for &(ref lint_name, level) in &sess.opts.lint_opts {
            store.check_lint_name_cmdline(sess, &lint_name, level);

            // If the cap is less than this specified level, e.g. if we've got
            // `--cap-lints allow` but we've also got `-D foo` then we ignore
            // this specification as the lint cap will set it to allow anyway.
            let level = cmp::min(level, self.lint_cap);

            let lint_flag_val = Symbol::intern(lint_name);
            let ids = match store.find_lints(&lint_name) {
                Ok(ids) => ids,
                Err(_) => continue, // errors handled in check_lint_name_cmdline above
            };
            for id in ids {
                let src = LintSource::CommandLine(lint_flag_val);
                specs.insert(id, (level, src));
            }
        }

        self.list.push(LintSet::CommandLine {
            specs: specs,
        });
    }

    fn get_lint_level(&self,
                      lint: &'static Lint,
                      idx: u32,
                      aux: Option<&FxHashMap<LintId, (Level, LintSource)>>)
        -> (Level, LintSource)
    {
        let (level, mut src) = self.get_lint_id_level(LintId::of(lint), idx, aux);

        // If `level` is none then we actually assume the default level for this
        // lint.
        let mut level = level.unwrap_or(lint.default_level);

        // If we're about to issue a warning, check at the last minute for any
        // directives against the warnings "lint". If, for example, there's an
        // `allow(warnings)` in scope then we want to respect that instead.
        if level == Level::Warn {
            let (warnings_level, warnings_src) =
                self.get_lint_id_level(LintId::of(lint::builtin::WARNINGS),
                                       idx,
                                       aux);
            if let Some(configured_warning_level) = warnings_level {
                if configured_warning_level != Level::Warn {
                    level = configured_warning_level;
                    src = warnings_src;
                }
            }
        }

        // Ensure that we never exceed the `--cap-lints` argument.
        level = cmp::min(level, self.lint_cap);

        return (level, src)
    }

    fn get_lint_id_level(&self,
                         id: LintId,
                         mut idx: u32,
                         aux: Option<&FxHashMap<LintId, (Level, LintSource)>>)
        -> (Option<Level>, LintSource)
    {
        if let Some(specs) = aux {
            if let Some(&(level, src)) = specs.get(&id) {
                return (Some(level), src)
            }
        }
        loop {
            match self.list[idx as usize] {
                LintSet::CommandLine { ref specs } => {
                    if let Some(&(level, src)) = specs.get(&id) {
                        return (Some(level), src)
                    }
                    return (None, LintSource::Default)
                }
                LintSet::Node { ref specs, parent } => {
                    if let Some(&(level, src)) = specs.get(&id) {
                        return (Some(level), src)
                    }
                    idx = parent;
                }
            }
        }
    }
}

pub struct LintLevelsBuilder<'a> {
    sess: &'a Session,
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, u32>,
    cur: u32,
    warn_about_weird_lints: bool,
}

pub struct BuilderPush {
    prev: u32,
}

impl<'a> LintLevelsBuilder<'a> {
    pub fn new(sess: &'a Session, sets: LintLevelSets) -> LintLevelsBuilder<'a> {
        assert_eq!(sets.list.len(), 1);
        LintLevelsBuilder {
            sess,
            sets,
            cur: 0,
            id_to_set: FxHashMap(),
            warn_about_weird_lints: sess.buffered_lints.borrow().is_some(),
        }
    }

    /// Pushes a list of AST lint attributes onto this context.
    ///
    /// This function will return a `BuilderPush` object which should be be
    /// passed to `pop` when this scope for the attributes provided is exited.
    ///
    /// This function will perform a number of tasks:
    ///
    /// * It'll validate all lint-related attributes in `attrs`
    /// * It'll mark all lint-related attriutes as used
    /// * Lint levels will be updated based on the attributes provided
    /// * Lint attributes are validated, e.g. a #[forbid] can't be switched to
    ///   #[allow]
    ///
    /// Don't forget to call `pop`!
    pub fn push(&mut self, attrs: &[ast::Attribute]) -> BuilderPush {
        let mut specs = FxHashMap();
        let store = self.sess.lint_store.borrow();
        let sess = self.sess;
        let bad_attr = |span| {
            span_err!(sess, span, E0452,
                      "malformed lint attribute");
        };
        for attr in attrs {
            let level = match attr.name().and_then(|name| Level::from_str(&name.as_str())) {
                None => continue,
                Some(lvl) => lvl,
            };

            let meta = unwrap_or!(attr.meta(), continue);
            attr::mark_used(attr);

            let metas = if let Some(metas) = meta.meta_item_list() {
                metas
            } else {
                bad_attr(meta.span);
                continue
            };

            for li in metas {
                let word = match li.word() {
                    Some(word) => word,
                    None => {
                        bad_attr(li.span);
                        continue
                    }
                };
                let name = word.name();
                match store.check_lint_name(&name.as_str()) {
                    CheckLintNameResult::Ok(ids) => {
                        let src = LintSource::Node(name, li.span);
                        for id in ids {
                            specs.insert(*id, (level, src));
                        }
                    }

                    _ if !self.warn_about_weird_lints => {}

                    CheckLintNameResult::Warning(ref msg) => {
                        let lint = builtin::RENAMED_AND_REMOVED_LINTS;
                        let (level, src) = self.sets.get_lint_level(lint,
                                                                    self.cur,
                                                                    Some(&specs));
                        lint::struct_lint_level(self.sess,
                                                lint,
                                                level,
                                                src,
                                                Some(li.span.into()),
                                                msg)
                            .emit();
                    }
                    CheckLintNameResult::NoLint => {
                        let lint = builtin::UNKNOWN_LINTS;
                        let (level, src) = self.sets.get_lint_level(lint,
                                                                    self.cur,
                                                                    Some(&specs));
                        let msg = format!("unknown lint: `{}`", name);
                        let mut db = lint::struct_lint_level(self.sess,
                                                lint,
                                                level,
                                                src,
                                                Some(li.span.into()),
                                                &msg);
                        if name.as_str().chars().any(|c| c.is_uppercase()) {
                            let name_lower = name.as_str().to_lowercase();
                            if let CheckLintNameResult::NoLint =
                                    store.check_lint_name(&name_lower) {
                                db.emit();
                            } else {
                                db.span_suggestion(
                                    li.span,
                                    "lowercase the lint name",
                                    name_lower
                                ).emit();
                            }
                        } else {
                            db.emit();
                        }
                    }
                }
            }
        }

        for (id, &(level, ref src)) in specs.iter() {
            if level == Level::Forbid {
                continue
            }
            let forbid_src = match self.sets.get_lint_id_level(*id, self.cur, None) {
                (Some(Level::Forbid), src) => src,
                _ => continue,
            };
            let forbidden_lint_name = match forbid_src {
                LintSource::Default => id.to_string(),
                LintSource::Node(name, _) => name.to_string(),
                LintSource::CommandLine(name) => name.to_string(),
            };
            let (lint_attr_name, lint_attr_span) = match *src {
                LintSource::Node(name, span) => (name, span),
                _ => continue,
            };
            let mut diag_builder = struct_span_err!(self.sess,
                                                    lint_attr_span,
                                                    E0453,
                                                    "{}({}) overruled by outer forbid({})",
                                                    level.as_str(),
                                                    lint_attr_name,
                                                    forbidden_lint_name);
            diag_builder.span_label(lint_attr_span, "overruled by previous forbid");
            match forbid_src {
                LintSource::Default => &mut diag_builder,
                LintSource::Node(_, forbid_source_span) => {
                    diag_builder.span_label(forbid_source_span,
                                            "`forbid` level set here")
                },
                LintSource::CommandLine(_) => {
                    diag_builder.note("`forbid` lint level was set on command line")
                }
            }.emit();
            // don't set a separate error for every lint in the group
            break
        }

        let prev = self.cur;
        if specs.len() > 0 {
            self.cur = self.sets.list.len() as u32;
            self.sets.list.push(LintSet::Node {
                specs: specs,
                parent: prev,
            });
        }

        BuilderPush {
            prev: prev,
        }
    }

    /// Called after `push` when the scope of a set of attributes are exited.
    pub fn pop(&mut self, push: BuilderPush) {
        self.cur = push.prev;
    }

    /// Used to emit a lint-related diagnostic based on the current state of
    /// this lint context.
    pub fn struct_lint(&self,
                       lint: &'static Lint,
                       span: Option<MultiSpan>,
                       msg: &str)
        -> DiagnosticBuilder<'a>
    {
        let (level, src) = self.sets.get_lint_level(lint, self.cur, None);
        lint::struct_lint_level(self.sess, lint, level, src, span, msg)
    }

    /// Registers the ID provided with the current set of lints stored in
    /// this context.
    pub fn register_id(&mut self, id: HirId) {
        self.id_to_set.insert(id, self.cur);
    }

    pub fn build(self) -> LintLevelSets {
        self.sets
    }

    pub fn build_map(self) -> LintLevelMap {
        LintLevelMap {
            sets: self.sets,
            id_to_set: self.id_to_set,
        }
    }
}

pub struct LintLevelMap {
    sets: LintLevelSets,
    id_to_set: FxHashMap<HirId, u32>,
}

impl LintLevelMap {
    /// If the `id` was previously registered with `register_id` when building
    /// this `LintLevelMap` this returns the corresponding lint level and source
    /// of the lint level for the lint provided.
    ///
    /// If the `id` was not previously registered, returns `None`. If `None` is
    /// returned then the parent of `id` should be acquired and this function
    /// should be called again.
    pub fn level_and_source(&self, lint: &'static Lint, id: HirId)
        -> Option<(Level, LintSource)>
    {
        self.id_to_set.get(&id).map(|idx| {
            self.sets.get_lint_level(lint, *idx, None)
        })
    }
}

impl<'gcx> HashStable<StableHashingContext<'gcx>> for LintLevelMap {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        let LintLevelMap {
            ref sets,
            ref id_to_set,
        } = *self;

        id_to_set.hash_stable(hcx, hasher);

        let LintLevelSets {
            ref list,
            lint_cap,
        } = *sets;

        lint_cap.hash_stable(hcx, hasher);

        hcx.while_hashing_spans(true, |hcx| {
            list.len().hash_stable(hcx, hasher);

            // We are working under the assumption here that the list of
            // lint-sets is built in a deterministic order.
            for lint_set in list {
                ::std::mem::discriminant(lint_set).hash_stable(hcx, hasher);

                match *lint_set {
                    LintSet::CommandLine { ref specs } => {
                        specs.hash_stable(hcx, hasher);
                    }
                    LintSet::Node { ref specs, parent } => {
                        specs.hash_stable(hcx, hasher);
                        parent.hash_stable(hcx, hasher);
                    }
                }
            }
        })
    }
}

impl<HCX> HashStable<HCX> for LintId {
    #[inline]
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut HCX,
                                          hasher: &mut StableHasher<W>) {
        self.lint_name_raw().hash_stable(hcx, hasher);
    }
}

impl<HCX> ToStableHashKey<HCX> for LintId {
    type KeyType = &'static str;

    #[inline]
    fn to_stable_hash_key(&self, _: &HCX) -> &'static str {
        self.lint_name_raw()
    }
}
