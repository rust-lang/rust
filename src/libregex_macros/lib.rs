// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate provides the `regex!` macro. Its use is documented in the
//! `regex` crate.

#![crate_id = "regex_macros#0.11.0"]
#![crate_type = "dylib"]
#![experimental]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]

#![feature(plugin_registrar, managed_boxes, quote)]

extern crate regex;
extern crate syntax;
extern crate rustc;

use std::rc::Rc;
use std::gc::{Gc, GC};

use syntax::ast;
use syntax::codemap;
use syntax::ext::build::AstBuilder;
use syntax::ext::base::{ExtCtxt, MacResult, MacExpr, DummyResult};
use syntax::parse::token;
use syntax::print::pprust;

use rustc::plugin::Registry;

use regex::Regex;
use regex::native::{
    OneChar, CharClass, Any, Save, Jump, Split,
    Match, EmptyBegin, EmptyEnd, EmptyWordBoundary,
    Program, Dynamic, Native,
    FLAG_NOCASE, FLAG_MULTI, FLAG_DOTNL, FLAG_NEGATED,
};

/// For the `regex!` syntax extension. Do not use.
#[plugin_registrar]
#[doc(hidden)]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("regex", native);
}

/// Generates specialized code for the Pike VM for a particular regular
/// expression.
///
/// There are two primary differences between the code generated here and the
/// general code in vm.rs.
///
/// 1. All heap allocation is removed. Sized vector types are used instead.
///    Care must be taken to make sure that these vectors are not copied
///    gratuitously. (If you're not sure, run the benchmarks. They will yell
///    at you if you do.)
/// 2. The main `match instruction { ... }` expressions are replaced with more
///    direct `match pc { ... }`. The generators can be found in
///    `step_insts` and `add_insts`.
///
/// Other more minor changes include eliding code when possible (although this
/// isn't completely thorough at the moment), and translating character class
/// matching from using a binary search to a simple `match` expression (see
/// `match_class`).
///
/// It is strongly recommended to read the dynamic implementation in vm.rs
/// first before trying to understand the code generator. The implementation
/// strategy is identical and vm.rs has comments and will be easier to follow.
#[allow(experimental)]
fn native(cx: &mut ExtCtxt, sp: codemap::Span, tts: &[ast::TokenTree])
          -> Box<MacResult> {
    let regex = match parse(cx, tts) {
        Some(r) => r,
        // error is logged in 'parse' with cx.span_err
        None => return DummyResult::any(sp),
    };
    let re = match Regex::new(regex.as_slice()) {
        Ok(re) => re,
        Err(err) => {
            cx.span_err(sp, err.to_str().as_slice());
            return DummyResult::any(sp)
        }
    };
    let prog = match re {
        Dynamic(Dynamic { ref prog, .. }) => prog.clone(),
        Native(_) => unreachable!(),
    };

    let mut gen = NfaGen {
        cx: &*cx, sp: sp, prog: prog,
        names: re.names_iter().collect(), original: re.as_str().to_string(),
    };
    MacExpr::new(gen.code())
}

struct NfaGen<'a> {
    cx: &'a ExtCtxt<'a>,
    sp: codemap::Span,
    prog: Program,
    names: Vec<Option<String>>,
    original: String,
}

impl<'a> NfaGen<'a> {
    fn code(&mut self) -> Gc<ast::Expr> {
        // Most or all of the following things are used in the quasiquoted
        // expression returned.
        let num_cap_locs = 2 * self.prog.num_captures();
        let num_insts = self.prog.insts.len();
        let cap_names = self.vec_expr(self.names.as_slice().iter(),
            |cx, name| match *name {
                Some(ref name) => {
                    let name = name.as_slice();
                    quote_expr!(cx, Some($name))
                }
                None => cx.expr_none(self.sp),
            }
        );
        let prefix_anchor =
            match self.prog.insts.as_slice()[1] {
                EmptyBegin(flags) if flags & FLAG_MULTI == 0 => true,
                _ => false,
            };
        let init_groups = self.vec_expr(range(0, num_cap_locs),
                                        |cx, _| cx.expr_none(self.sp));

        let prefix_lit = Rc::new(Vec::from_slice(self.prog.prefix.as_slice().as_bytes()));
        let prefix_bytes = self.cx.expr_lit(self.sp, ast::LitBinary(prefix_lit));

        let check_prefix = self.check_prefix();
        let step_insts = self.step_insts();
        let add_insts = self.add_insts();
        let regex = self.original.as_slice();

        quote_expr!(self.cx, {
// When `regex!` is bound to a name that is not used, we have to make sure
// that dead_code warnings don't bubble up to the user from the generated
// code. Therefore, we suppress them by allowing dead_code. The effect is that
// the user is only warned about *their* unused variable/code, and not the
// unused code generated by regex!. See #14185 for an example.
#[allow(dead_code)]
static CAP_NAMES: &'static [Option<&'static str>] = &$cap_names;

#[allow(dead_code)]
fn exec<'t>(which: ::regex::native::MatchKind, input: &'t str,
            start: uint, end: uint) -> Vec<Option<uint>> {
    #![allow(unused_imports)]
    #![allow(unused_mut)]

    use regex::native::{
        MatchKind, Exists, Location, Submatches,
        StepState, StepMatchEarlyReturn, StepMatch, StepContinue,
        CharReader, find_prefix,
    };

    return Nfa {
        which: which,
        input: input,
        ic: 0,
        chars: CharReader::new(input),
    }.run(start, end);

    type Captures = [Option<uint>, ..$num_cap_locs];

    struct Nfa<'t> {
        which: MatchKind,
        input: &'t str,
        ic: uint,
        chars: CharReader<'t>,
    }

    impl<'t> Nfa<'t> {
        #[allow(unused_variable)]
        fn run(&mut self, start: uint, end: uint) -> Vec<Option<uint>> {
            let mut matched = false;
            let prefix_bytes: &[u8] = $prefix_bytes;
            let mut clist = &mut Threads::new(self.which);
            let mut nlist = &mut Threads::new(self.which);

            let mut groups = $init_groups;

            self.ic = start;
            let mut next_ic = self.chars.set(start);
            while self.ic <= end {
                if clist.size == 0 {
                    if matched {
                        break
                    }
                    $check_prefix
                }
                if clist.size == 0 || (!$prefix_anchor && !matched) {
                    self.add(clist, 0, &mut groups)
                }

                self.ic = next_ic;
                next_ic = self.chars.advance();

                for i in range(0, clist.size) {
                    let pc = clist.pc(i);
                    let step_state = self.step(&mut groups, nlist,
                                               clist.groups(i), pc);
                    match step_state {
                        StepMatchEarlyReturn =>
                            return vec![Some(0u), Some(0u)],
                        StepMatch => { matched = true; break },
                        StepContinue => {},
                    }
                }
                ::std::mem::swap(&mut clist, &mut nlist);
                nlist.empty();
            }
            match self.which {
                Exists if matched     => vec![Some(0u), Some(0u)],
                Exists                => vec![None, None],
                Location | Submatches => groups.iter().map(|x| *x).collect(),
            }
        }

        // Sometimes `nlist` is never used (for empty regexes).
        #[allow(unused_variable)]
        #[inline]
        fn step(&self, groups: &mut Captures, nlist: &mut Threads,
                caps: &mut Captures, pc: uint) -> StepState {
            $step_insts
            StepContinue
        }

        fn add(&self, nlist: &mut Threads, pc: uint,
               groups: &mut Captures) {
            if nlist.contains(pc) {
                return
            }
            $add_insts
        }
    }

    struct Thread {
        pc: uint,
        groups: Captures,
    }

    struct Threads {
        which: MatchKind,
        queue: [Thread, ..$num_insts],
        sparse: [uint, ..$num_insts],
        size: uint,
    }

    impl Threads {
        fn new(which: MatchKind) -> Threads {
            Threads {
                which: which,
                // These unsafe blocks are used for performance reasons, as it
                // gives us a zero-cost initialization of a sparse set. The
                // trick is described in more detail here:
                // http://research.swtch.com/sparse
                // The idea here is to avoid initializing threads that never
                // need to be initialized, particularly for larger regexs with
                // a lot of instructions.
                queue: unsafe { ::std::mem::uninitialized() },
                sparse: unsafe { ::std::mem::uninitialized() },
                size: 0,
            }
        }

        #[inline]
        fn add(&mut self, pc: uint, groups: &Captures) {
            let t = &mut self.queue[self.size];
            t.pc = pc;
            match self.which {
                Exists => {},
                Location => {
                    t.groups[0] = groups[0];
                    t.groups[1] = groups[1];
                }
                Submatches => {
                    for (slot, val) in t.groups.mut_iter().zip(groups.iter()) {
                        *slot = *val;
                    }
                }
            }
            self.sparse[pc] = self.size;
            self.size += 1;
        }

        #[inline]
        fn add_empty(&mut self, pc: uint) {
            self.queue[self.size].pc = pc;
            self.sparse[pc] = self.size;
            self.size += 1;
        }

        #[inline]
        fn contains(&self, pc: uint) -> bool {
            let s = self.sparse[pc];
            s < self.size && self.queue[s].pc == pc
        }

        #[inline]
        fn empty(&mut self) {
            self.size = 0;
        }

        #[inline]
        fn pc(&self, i: uint) -> uint {
            self.queue[i].pc
        }

        #[inline]
        fn groups<'r>(&'r mut self, i: uint) -> &'r mut Captures {
            &'r mut self.queue[i].groups
        }
    }
}

::regex::native::Native(::regex::native::Native {
    original: $regex,
    names: CAP_NAMES,
    prog: exec,
})
        })
    }

    // Generates code for the `add` method, which is responsible for adding
    // zero-width states to the next queue of states to visit.
    fn add_insts(&self) -> Gc<ast::Expr> {
        let arms = self.prog.insts.iter().enumerate().map(|(pc, inst)| {
            let nextpc = pc + 1;
            let body = match *inst {
                EmptyBegin(flags) => {
                    let cond =
                        if flags & FLAG_MULTI > 0 {
                            quote_expr!(self.cx,
                                self.chars.is_begin()
                                || self.chars.prev == Some('\n')
                            )
                        } else {
                            quote_expr!(self.cx, self.chars.is_begin())
                        };
                    quote_expr!(self.cx, {
                        nlist.add_empty($pc);
                        if $cond { self.add(nlist, $nextpc, &mut *groups) }
                    })
                }
                EmptyEnd(flags) => {
                    let cond =
                        if flags & FLAG_MULTI > 0 {
                            quote_expr!(self.cx,
                                self.chars.is_end()
                                || self.chars.cur == Some('\n')
                            )
                        } else {
                            quote_expr!(self.cx, self.chars.is_end())
                        };
                    quote_expr!(self.cx, {
                        nlist.add_empty($pc);
                        if $cond { self.add(nlist, $nextpc, &mut *groups) }
                    })
                }
                EmptyWordBoundary(flags) => {
                    let cond =
                        if flags & FLAG_NEGATED > 0 {
                            quote_expr!(self.cx, !self.chars.is_word_boundary())
                        } else {
                            quote_expr!(self.cx, self.chars.is_word_boundary())
                        };
                    quote_expr!(self.cx, {
                        nlist.add_empty($pc);
                        if $cond { self.add(nlist, $nextpc, &mut *groups) }
                    })
                }
                Save(slot) => {
                    let save = quote_expr!(self.cx, {
                        let old = groups[$slot];
                        groups[$slot] = Some(self.ic);
                        self.add(nlist, $nextpc, &mut *groups);
                        groups[$slot] = old;
                    });
                    let add = quote_expr!(self.cx, {
                        self.add(nlist, $nextpc, &mut *groups);
                    });
                    // If this is saving a submatch location but we request
                    // existence or only full match location, then we can skip
                    // right over it every time.
                    if slot > 1 {
                        quote_expr!(self.cx, {
                            nlist.add_empty($pc);
                            match self.which {
                                Submatches => $save,
                                Exists | Location => $add,
                            }
                        })
                    } else {
                        quote_expr!(self.cx, {
                            nlist.add_empty($pc);
                            match self.which {
                                Submatches | Location => $save,
                                Exists => $add,
                            }
                        })
                    }
                }
                Jump(to) => {
                    quote_expr!(self.cx, {
                        nlist.add_empty($pc);
                        self.add(nlist, $to, &mut *groups);
                    })
                }
                Split(x, y) => {
                    quote_expr!(self.cx, {
                        nlist.add_empty($pc);
                        self.add(nlist, $x, &mut *groups);
                        self.add(nlist, $y, &mut *groups);
                    })
                }
                // For Match, OneChar, CharClass, Any
                _ => quote_expr!(self.cx, nlist.add($pc, &*groups)),
            };
            self.arm_inst(pc, body)
        }).collect::<Vec<ast::Arm>>();

        self.match_insts(arms)
    }

    // Generates the code for the `step` method, which processes all states
    // in the current queue that consume a single character.
    fn step_insts(&self) -> Gc<ast::Expr> {
        let arms = self.prog.insts.iter().enumerate().map(|(pc, inst)| {
            let nextpc = pc + 1;
            let body = match *inst {
                Match => {
                    quote_expr!(self.cx, {
                        match self.which {
                            Exists => {
                                return StepMatchEarlyReturn
                            }
                            Location => {
                                groups[0] = caps[0];
                                groups[1] = caps[1];
                                return StepMatch
                            }
                            Submatches => {
                                for (slot, val) in groups.mut_iter().zip(caps.iter()) {
                                    *slot = *val;
                                }
                                return StepMatch
                            }
                        }
                    })
                }
                OneChar(c, flags) => {
                    if flags & FLAG_NOCASE > 0 {
                        let upc = c.to_uppercase();
                        quote_expr!(self.cx, {
                            let upc = self.chars.prev.map(|c| c.to_uppercase());
                            if upc == Some($upc) {
                                self.add(nlist, $nextpc, caps);
                            }
                        })
                    } else {
                        quote_expr!(self.cx, {
                            if self.chars.prev == Some($c) {
                                self.add(nlist, $nextpc, caps);
                            }
                        })
                    }
                }
                CharClass(ref ranges, flags) => {
                    let negate = flags & FLAG_NEGATED > 0;
                    let casei = flags & FLAG_NOCASE > 0;
                    let get_char =
                        if casei {
                            quote_expr!(self.cx, self.chars.prev.unwrap().to_uppercase())
                        } else {
                            quote_expr!(self.cx, self.chars.prev.unwrap())
                        };
                    let negcond =
                        if negate {
                            quote_expr!(self.cx, !found)
                        } else {
                            quote_expr!(self.cx, found)
                        };
                    let mranges = self.match_class(casei, ranges.as_slice());
                    quote_expr!(self.cx, {
                        if self.chars.prev.is_some() {
                            let c = $get_char;
                            let found = $mranges;
                            if $negcond {
                                self.add(nlist, $nextpc, caps);
                            }
                        }
                    })
                }
                Any(flags) => {
                    if flags & FLAG_DOTNL > 0 {
                        quote_expr!(self.cx, self.add(nlist, $nextpc, caps))
                    } else {
                        quote_expr!(self.cx, {
                            if self.chars.prev != Some('\n') {
                                self.add(nlist, $nextpc, caps)
                            }
                            ()
                        })
                    }
                }
                // EmptyBegin, EmptyEnd, EmptyWordBoundary, Save, Jump, Split
                _ => self.empty_block(),
            };
            self.arm_inst(pc, body)
        }).collect::<Vec<ast::Arm>>();

        self.match_insts(arms)
    }

    // Translates a character class into a match expression.
    // This avoids a binary search (and is hopefully replaced by a jump
    // table).
    fn match_class(&self, casei: bool, ranges: &[(char, char)]) -> Gc<ast::Expr> {
        let expr_true = quote_expr!(self.cx, true);

        let mut arms = ranges.iter().map(|&(mut start, mut end)| {
            if casei {
                start = start.to_uppercase();
                end = end.to_uppercase();
            }
            let pat = self.cx.pat(self.sp, ast::PatRange(quote_expr!(self.cx, $start),
                                                         quote_expr!(self.cx, $end)));
            self.cx.arm(self.sp, vec!(pat), expr_true)
        }).collect::<Vec<ast::Arm>>();

        arms.push(self.wild_arm_expr(quote_expr!(self.cx, false)));

        let match_on = quote_expr!(self.cx, c);
        self.cx.expr_match(self.sp, match_on, arms)
    }

    // Generates code for checking a literal prefix of the search string.
    // The code is only generated if the regex *has* a literal prefix.
    // Otherwise, a no-op is returned.
    fn check_prefix(&self) -> Gc<ast::Expr> {
        if self.prog.prefix.len() == 0 {
            self.empty_block()
        } else {
            quote_expr!(self.cx,
                if clist.size == 0 {
                    let haystack = self.input.as_bytes().slice_from(self.ic);
                    match find_prefix(prefix_bytes, haystack) {
                        None => break,
                        Some(i) => {
                            self.ic += i;
                            next_ic = self.chars.set(self.ic);
                        }
                    }
                }
            )
        }
    }

    // Builds a `match pc { ... }` expression from a list of arms, specifically
    // for matching the current program counter with an instruction.
    // A wild-card arm is automatically added that executes a no-op. It will
    // never be used, but is added to satisfy the compiler complaining about
    // non-exhaustive patterns.
    fn match_insts(&self, mut arms: Vec<ast::Arm>) -> Gc<ast::Expr> {
        arms.push(self.wild_arm_expr(self.empty_block()));
        self.cx.expr_match(self.sp, quote_expr!(self.cx, pc), arms)
    }

    fn empty_block(&self) -> Gc<ast::Expr> {
        quote_expr!(self.cx, {})
    }

    // Creates a match arm for the instruction at `pc` with the expression
    // `body`.
    fn arm_inst(&self, pc: uint, body: Gc<ast::Expr>) -> ast::Arm {
        let pc_pat = self.cx.pat_lit(self.sp, quote_expr!(self.cx, $pc));

        self.cx.arm(self.sp, vec!(pc_pat), body)
    }

    // Creates a wild-card match arm with the expression `body`.
    fn wild_arm_expr(&self, body: Gc<ast::Expr>) -> ast::Arm {
        ast::Arm {
            attrs: vec!(),
            pats: vec!(box(GC) ast::Pat{
                id: ast::DUMMY_NODE_ID,
                span: self.sp,
                node: ast::PatWild,
            }),
            guard: None,
            body: body,
        }
    }


    // Converts `xs` to a `[x1, x2, .., xN]` expression by calling `to_expr`
    // on each element in `xs`.
    fn vec_expr<T, It: Iterator<T>>(&self, xs: It,
                                    to_expr: |&ExtCtxt, T| -> Gc<ast::Expr>)
                  -> Gc<ast::Expr> {
        let exprs = xs.map(|x| to_expr(self.cx, x)).collect();
        self.cx.expr_vec(self.sp, exprs)
    }
}

/// Looks for a single string literal and returns it.
/// Otherwise, logs an error with cx.span_err and returns None.
fn parse(cx: &mut ExtCtxt, tts: &[ast::TokenTree]) -> Option<String> {
    let mut parser = cx.new_parser_from_tts(tts);
    let entry = cx.expand_expr(parser.parse_expr());
    let regex = match entry.node {
        ast::ExprLit(lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => s.to_str(),
                _ => {
                    cx.span_err(entry.span, format!(
                        "expected string literal but got `{}`",
                        pprust::lit_to_str(lit)).as_slice());
                    return None
                }
            }
        }
        _ => {
            cx.span_err(entry.span, format!(
                "expected string literal but got `{}`",
                pprust::expr_to_str(entry)).as_slice());
            return None
        }
    };
    if !parser.eat(&token::EOF) {
        cx.span_err(parser.span, "only one string literal allowed");
        return None;
    }
    Some(regex)
}
