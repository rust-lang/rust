use {CrateLint, PathResult, Segment};
use macros::ParentScope;

use syntax::symbol::keywords;
use syntax_pos::Span;

use resolve_imports::ImportResolver;
use std::cmp::Reverse;

impl<'a, 'b:'a> ImportResolver<'a, 'b> {
    /// Add suggestions for a path that cannot be resolved.
    pub(crate) fn make_path_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        debug!("make_path_suggestion: span={:?} path={:?}", span, path);

        match (path.get(0), path.get(1)) {
            // `{{root}}::ident::...` on both editions.
            // On 2015 `{{root}}` is usually added implicitly.
            (Some(fst), Some(snd)) if fst.ident.name == keywords::PathRoot.name() &&
                                      !snd.ident.is_path_segment_keyword() => {}
            // `ident::...` on 2018
            (Some(fst), _) if fst.ident.span.rust_2018() &&
                              !fst.ident.is_path_segment_keyword() => {
                // Insert a placeholder that's later replaced by `self`/`super`/etc.
                path.insert(0, Segment::from_ident(keywords::Invalid.ident()));
            }
            _ => return None,
        }

        self.make_missing_self_suggestion(span, path.clone(), parent_scope)
            .or_else(|| self.make_missing_crate_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_missing_super_suggestion(span, path.clone(), parent_scope))
            .or_else(|| self.make_external_crate_suggestion(span, path, parent_scope))
    }

    /// Suggest a missing `self::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `self::foo`?
    /// ```
    fn make_missing_self_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `self` and check if that is valid.
        path[0].ident.name = keywords::SelfLower.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_self_suggestion: path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, None))
        } else {
            None
        }
    }

    /// Suggest a missing `crate::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `crate::foo`?
    /// ```
    fn make_missing_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = keywords::Crate.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_crate_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((
                path,
                Some(
                    "`use` statements changed in Rust 2018; read more at \
                     <https://doc.rust-lang.org/edition-guide/rust-2018/module-system/path-\
                     clarity.html>".to_string()
                ),
            ))
        } else {
            None
        }
    }

    /// Suggest a missing `super::` if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foo::Bar;
    ///    |     ^^^ did you mean `super::foo`?
    /// ```
    fn make_missing_super_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        // Replace first ident with `crate` and check if that is valid.
        path[0].ident.name = keywords::Super.name();
        let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
        debug!("make_missing_super_suggestion:  path={:?} result={:?}", path, result);
        if let PathResult::Module(..) = result {
            Some((path, None))
        } else {
            None
        }
    }

    /// Suggest a missing external crate name if that resolves to an correct module.
    ///
    /// ```
    ///    |
    /// LL | use foobar::Baz;
    ///    |     ^^^^^^ did you mean `baz::foobar`?
    /// ```
    ///
    /// Used when importing a submodule of an external crate but missing that crate's
    /// name as the first part of path.
    fn make_external_crate_suggestion(
        &mut self,
        span: Span,
        mut path: Vec<Segment>,
        parent_scope: &ParentScope<'b>,
    ) -> Option<(Vec<Segment>, Option<String>)> {
        if path[1].ident.span.rust_2015() {
            return None;
        }

        // Sort extern crate names in reverse order to get
        // 1) some consistent ordering for emitted dignostics and
        // 2) `std` suggestions before `core` suggestions.
        let mut extern_crate_names =
            self.resolver.extern_prelude.iter().map(|(ident, _)| ident.name).collect::<Vec<_>>();
        extern_crate_names.sort_by_key(|name| Reverse(name.as_str()));

        for name in extern_crate_names.into_iter() {
            // Replace first ident with a crate name and check if that is valid.
            path[0].ident.name = name;
            let result = self.resolve_path(&path, None, parent_scope, false, span, CrateLint::No);
            debug!("make_external_crate_suggestion: name={:?} path={:?} result={:?}",
                    name, path, result);
            if let PathResult::Module(..) = result {
                return Some((path, None));
            }
        }

        None
    }
}
