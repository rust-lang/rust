# Inline assembly

## Overview

Inline assembly in rustc mostly revolves around taking an `asm!` macro invocation and plumbing it
through all of the compiler layers down to LLVM codegen. Throughout the various stages, an
`InlineAsm` generally consists of 3 components:

- The template string, which is stored as an array of `InlineAsmTemplatePiece`. Each piece
represents either a literal or a placeholder for an operand (just like format strings).

  ```rust
  pub enum InlineAsmTemplatePiece {
      String(String),
      Placeholder { operand_idx: usize, modifier: Option<char>, span: Span },
  }
  ```

- The list of operands to the `asm!` (`in`, `[late]out`, `in[late]out`, `sym`, `const`). These are
represented differently at each stage of lowering, but follow a common pattern:
  - `in`, `out` and `inout` all have an associated register class (`reg`) or explicit register
(`"eax"`).
  - `inout` has 2 forms: one with a single expression that is both read from and written to, and
one with two separate expressions for the input and output parts.
  - `out` and `inout` have a `late` flag (`lateout` / `inlateout`) to indicate that the register
allocator is allowed to reuse an input register for this output.
  - `out` and the split variant of `inout` allow `_` to be specified for an output, which means
that the output is discarded. This is used to allocate scratch registers for assembly code.
  - `const` refers to an anonymous constants and generally works like an inline const.
  - `sym` is a bit special since it only accepts a path expression, which must point to a `static`
or a `fn`.

- The options set at the end of the `asm!` macro. The only ones that are of particular interest to
rustc are `NORETURN` which makes `asm!` return `!` instead of `()`, and `RAW` which disables format
string parsing. The remaining options are mostly passed through to LLVM with little processing.

  ```rust
  bitflags::bitflags! {
      pub struct InlineAsmOptions: u16 {
          const PURE = 1 << 0;
          const NOMEM = 1 << 1;
          const READONLY = 1 << 2;
          const PRESERVES_FLAGS = 1 << 3;
          const NORETURN = 1 << 4;
          const NOSTACK = 1 << 5;
          const ATT_SYNTAX = 1 << 6;
          const RAW = 1 << 7;
          const MAY_UNWIND = 1 << 8;
      }
  }
  ```

## AST

`InlineAsm` is represented as an expression in the AST with the [`ast::InlineAsm` type][inline_asm_ast].

The `asm!` macro is implemented in `rustc_builtin_macros` and outputs an `InlineAsm` AST node. The
template string is parsed using `fmt_macros`, positional and named operands are resolved to
explicit operand indices. Since target information is not available to macro invocations,
validation of the registers and register classes is deferred to AST lowering.

[inline_asm_ast]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/struct.InlineAsm.html

## HIR

`InlineAsm` is represented as an expression in the HIR with the [`hir::InlineAsm` type][inline_asm_hir].

AST lowering is where `InlineAsmRegOrRegClass` is converted from `Symbol`s to an actual register or
register class. If any modifiers are specified for a template string placeholder, these are
validated against the set allowed for that operand type. Finally, explicit registers for inputs and
outputs are checked for conflicts (same register used for different operands).

[inline_asm_hir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/struct.InlineAsm.html

## Type checking

Each register class has a whitelist of types that it may be used with. After the types of all
operands have been determined, the `intrinsicck` pass will check that these types are in the
whitelist. It also checks that split `inout` operands have compatible types and that `const`
operands are integers or floats. Suggestions are emitted where needed if a template modifier should
be used for an operand based on the type that was passed into it.

## THIR

`InlineAsm` is represented as an expression in the THIR with the [`InlineAsmExpr` type][inline_asm_thir].

The only significant change compared to HIR is that `Sym` has been lowered to either a `SymFn`
whose `expr` is a `Literal` ZST of the `fn`, or a `SymStatic` which points to the `DefId` of a
`static`.

[inline_asm_thir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/thir/struct.InlineAsmExpr.html

## MIR

`InlineAsm` is represented as a `Terminator` in the MIR with the [`TerminatorKind::InlineAsm` variant][inline_asm_mir]

As part of THIR lowering, `InOut` and `SplitInOut` operands are lowered to a split form with a
separate `in_value` and `out_place`.

Semantically, the `InlineAsm` terminator is similar to the `Call` terminator except that it has
multiple output places where a `Call` only has a single return place output.

[inline_asm_mir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.TerminatorKind.html#variant.InlineAsm

## Codegen

Operands are lowered one more time before being passed to LLVM codegen, this is represented by the [`InlineAsmOperandRef` type][inline_asm_codegen] from `rustc_codegen_ssa`.

The operands are lowered to LLVM operands and constraint codes as follows:
- `out` and the output part of `inout` operands are added first, as required by LLVM. Late output
operands have a `=` prefix added to their constraint code, non-late output operands have a `=&`
prefix added to their constraint code.
- `in` operands are added normally.
- `inout` operands are tied to the matching output operand.
- `sym` operands are passed as function pointers or pointers, using the `"s"` constraint.
- `const` operands are formatted to a string and directly inserted in the template string.

The template string is converted to LLVM form:
- `$` characters are escaped as `$$`.
- `const` operands are converted to strings and inserted directly.
- Placeholders are formatted as `${X:M}` where `X` is the operand index and `M` is the modifier
character. Modifiers are converted from the Rust form to the LLVM form.

The various options are converted to clobber constraints or LLVM attributes, refer to the
[RFC](https://github.com/Amanieu/rfcs/blob/inline-asm/text/0000-inline-asm.md#mapping-to-llvm-ir)
for more details.

Note that LLVM is sometimes rather picky about what types it accepts for certain constraint codes
so we sometimes need to insert conversions to/from a supported type. See the target-specific
ISelLowering.cpp files in LLVM for details of what types are supported for each register class.

[inline_asm_codegen]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/traits/enum.InlineAsmOperandRef.html

## Adding support for new architectures

Adding inline assembly support to an architecture is mostly a matter of defining the registers and
register classes for that architecture. All the definitions for register classes are located in
`compiler/rustc_target/asm/`.

Additionally you will need to implement lowering of these register classes to LLVM constraint codes
in `compiler/rustc_codegen_llvm/asm.rs`.

When adding a new architecture, make sure to cross-reference with the LLVM source code:
- LLVM has restrictions on which types can be used with a particular constraint code. Refer to the
`getRegForInlineAsmConstraint` function in `lib/Target/${ARCH}/${ARCH}ISelLowering.cpp`.
- LLVM reserves certain registers for its internal use, which causes them to not be saved/restored
properly around inline assembly blocks. These registers are listed in the `getReservedRegs`
function in `lib/Target/${ARCH}/${ARCH}RegisterInfo.cpp`. Any "conditionally" reserved register
such as the frame/base pointer must always be treated as reserved for Rust purposes because we
can't know ahead of time whether a function will require a frame/base pointer.

## Tests

Various tests for inline assembly are available:

- `tests/assembly-llvm/asm`
- `tests/ui/asm`
- `tests/codegen-llvm/asm-*`

Every architecture supported by inline assembly must have exhaustive tests in
`tests/assembly-llvm/asm` which test all combinations of register classes and types.
