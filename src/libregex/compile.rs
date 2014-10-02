// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Enable this to squash warnings due to exporting pieces of the representation
// for use with the regex! macro. See lib.rs for explanation.

use std::cmp;
use parse;
use parse::{
    Flags, FLAG_EMPTY,
    Nothing, Literal, Dot, AstClass, Begin, End, WordBoundary, Capture, Cat, Alt,
    Rep,
    ZeroOne, ZeroMore, OneMore,
};

type InstIdx = uint;

#[deriving(Show, Clone)]
pub enum Inst {
    // When a Match instruction is executed, the current thread is successful.
    Match,

    // The OneChar instruction matches a literal character.
    // The flags indicate whether to do a case insensitive match.
    OneChar(char, Flags),

    // The CharClass instruction tries to match one input character against
    // the range of characters given.
    // The flags indicate whether to do a case insensitive match and whether
    // the character class is negated or not.
    CharClass(Vec<(char, char)>, Flags),

    // Matches any character except new lines.
    // The flags indicate whether to include the '\n' character.
    Any(Flags),

    // Matches the beginning of the string, consumes no characters.
    // The flags indicate whether it matches if the preceding character
    // is a new line.
    EmptyBegin(Flags),

    // Matches the end of the string, consumes no characters.
    // The flags indicate whether it matches if the proceeding character
    // is a new line.
    EmptyEnd(Flags),

    // Matches a word boundary (\w on one side and \W \A or \z on the other),
    // and consumes no character.
    // The flags indicate whether this matches a word boundary or something
    // that isn't a word boundary.
    EmptyWordBoundary(Flags),

    // Saves the current position in the input string to the Nth save slot.
    Save(uint),

    // Jumps to the instruction at the index given.
    Jump(InstIdx),

    // Jumps to the instruction at the first index given. If that leads to
    // a failing state, then the instruction at the second index given is
    // tried.
    Split(InstIdx, InstIdx),
}

/// Program represents a compiled regular expression. Once an expression is
/// compiled, its representation is immutable and will never change.
///
/// All of the data in a compiled expression is wrapped in "MaybeStatic" or
/// "MaybeOwned" types so that a `Program` can be represented as static data.
/// (This makes it convenient and efficient for use with the `regex!` macro.)
#[deriving(Clone)]
pub struct Program {
    /// A sequence of instructions.
    pub insts: Vec<Inst>,
    /// If the regular expression requires a literal prefix in order to have a
    /// match, that prefix is stored here. (It's used in the VM to implement
    /// an optimization.)
    pub prefix: String,
}

impl Program {
    /// Compiles a Regex given its AST.
    pub fn new(ast: parse::Ast) -> (Program, Vec<Option<String>>) {
        let mut c = Compiler {
            insts: Vec::with_capacity(100),
            names: Vec::with_capacity(10),
        };

        c.insts.push(Save(0));
        c.compile(ast);
        c.insts.push(Save(1));
        c.insts.push(Match);

        // Try to discover a literal string prefix.
        // This is a bit hacky since we have to skip over the initial
        // 'Save' instruction.
        let mut pre = String::with_capacity(5);
        for inst in c.insts.slice_from(1).iter() {
            match *inst {
                OneChar(c, FLAG_EMPTY) => pre.push(c),
                _ => break
            }
        }

        let Compiler { insts, names } = c;
        let prog = Program {
            insts: insts,
            prefix: pre,
        };
        (prog, names)
    }

    /// Returns the total number of capture groups in the regular expression.
    /// This includes the zeroth capture.
    pub fn num_captures(&self) -> uint {
        let mut n = 0;
        for inst in self.insts.iter() {
            match *inst {
                Save(c) => n = cmp::max(n, c+1),
                _ => {}
            }
        }
        // There's exactly 2 Save slots for every capture.
        n / 2
    }
}

struct Compiler<'r> {
    insts: Vec<Inst>,
    names: Vec<Option<String>>,
}

// The compiler implemented here is extremely simple. Most of the complexity
// in this crate is in the parser or the VM.
// The only tricky thing here is patching jump/split instructions to point to
// the right instruction.
impl<'r> Compiler<'r> {
    fn compile(&mut self, ast: parse::Ast) {
        match ast {
            Nothing => {},
            Literal(c, flags) => self.push(OneChar(c, flags)),
            Dot(nl) => self.push(Any(nl)),
            AstClass(ranges, flags) =>
                self.push(CharClass(ranges, flags)),
            Begin(flags) => self.push(EmptyBegin(flags)),
            End(flags) => self.push(EmptyEnd(flags)),
            WordBoundary(flags) => self.push(EmptyWordBoundary(flags)),
            Capture(cap, name, x) => {
                let len = self.names.len();
                if cap >= len {
                    self.names.grow(10 + cap - len, None)
                }
                *self.names.get_mut(cap) = name;

                self.push(Save(2 * cap));
                self.compile(*x);
                self.push(Save(2 * cap + 1));
            }
            Cat(xs) => {
                for x in xs.into_iter() {
                    self.compile(x)
                }
            }
            Alt(x, y) => {
                let split = self.empty_split(); // push: split 0, 0
                let j1 = self.insts.len();
                self.compile(*x);                // push: insts for x
                let jmp = self.empty_jump();    // push: jmp 0
                let j2 = self.insts.len();
                self.compile(*y);                // push: insts for y
                let j3 = self.insts.len();

                self.set_split(split, j1, j2);  // split 0, 0 -> split j1, j2
                self.set_jump(jmp, j3);         // jmp 0      -> jmp j3
            }
            Rep(x, ZeroOne, g) => {
                let split = self.empty_split();
                let j1 = self.insts.len();
                self.compile(*x);
                let j2 = self.insts.len();

                if g.is_greedy() {
                    self.set_split(split, j1, j2);
                } else {
                    self.set_split(split, j2, j1);
                }
            }
            Rep(x, ZeroMore, g) => {
                let j1 = self.insts.len();
                let split = self.empty_split();
                let j2 = self.insts.len();
                self.compile(*x);
                let jmp = self.empty_jump();
                let j3 = self.insts.len();

                self.set_jump(jmp, j1);
                if g.is_greedy() {
                    self.set_split(split, j2, j3);
                } else {
                    self.set_split(split, j3, j2);
                }
            }
            Rep(x, OneMore, g) => {
                let j1 = self.insts.len();
                self.compile(*x);
                let split = self.empty_split();
                let j2 = self.insts.len();

                if g.is_greedy() {
                    self.set_split(split, j1, j2);
                } else {
                    self.set_split(split, j2, j1);
                }
            }
        }
    }

    /// Appends the given instruction to the program.
    #[inline]
    fn push(&mut self, x: Inst) {
        self.insts.push(x)
    }

    /// Appends an *empty* `Split` instruction to the program and returns
    /// the index of that instruction. (The index can then be used to "patch"
    /// the actual locations of the split in later.)
    #[inline]
    fn empty_split(&mut self) -> InstIdx {
        self.insts.push(Split(0, 0));
        self.insts.len() - 1
    }

    /// Sets the left and right locations of a `Split` instruction at index
    /// `i` to `pc1` and `pc2`, respectively.
    /// If the instruction at index `i` isn't a `Split` instruction, then
    /// `fail!` is called.
    #[inline]
    fn set_split(&mut self, i: InstIdx, pc1: InstIdx, pc2: InstIdx) {
        let split = self.insts.get_mut(i);
        match *split {
            Split(_, _) => *split = Split(pc1, pc2),
            _ => fail!("BUG: Invalid split index."),
        }
    }

    /// Appends an *empty* `Jump` instruction to the program and returns the
    /// index of that instruction.
    #[inline]
    fn empty_jump(&mut self) -> InstIdx {
        self.insts.push(Jump(0));
        self.insts.len() - 1
    }

    /// Sets the location of a `Jump` instruction at index `i` to `pc`.
    /// If the instruction at index `i` isn't a `Jump` instruction, then
    /// `fail!` is called.
    #[inline]
    fn set_jump(&mut self, i: InstIdx, pc: InstIdx) {
        let jmp = self.insts.get_mut(i);
        match *jmp {
            Jump(_) => *jmp = Jump(pc),
            _ => fail!("BUG: Invalid jump index."),
        }
    }
}
