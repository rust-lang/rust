% Syntax Index

## Keywords

* `as`: primitive casting.  See [Casting Between Types (`as`)].
* `break`: break out of loop.  See [Loops (Ending Iteration Early)].
* `const`: constant items.  See [`const` and `static`].
* `continue`: continue to next loop iteration.  See [Loops (Ending Iteration Early)].
* `crate`: external crate linkage.  See [Crates and Modules (Importing External Crates)].
* `else`: fallback for `if` and `if let` constructs.  See [`if`], [`if let`].
* `enum`: defining enumeration.  See [Enums].
* `extern`: external crate, function, and variable linkage.  See [Crates and Modules (Importing External Crates)], [Foreign Function Interface].
* `false`: boolean false literal.  See [Primitive Types (Booleans)].
* `fn`: function definition and function pointer types.  See [Functions].
* `for`: iterator loop, part of trait `impl` syntax, and higher-ranked lifetime syntax.  See [Loops (`for`)], [Method Syntax].
* `if`: conditional branching.  See [`if`], [`if let`].
* `impl`: inherent and trait implementation blocks.  See [Method Syntax].
* `in`: part of `for` loop syntax.  See [Loops (`for`)].
* `let`: variable binding.  See [Variable Bindings].
* `loop`: unconditional, infinite loop.  See [Loops (`loop`)].
* `match`: pattern matching.  See [Match].
* `mod`: module declaration.  See [Crates and Modules (Defining Modules)].
* `move`: part of closure syntax.  See [Closures (`move` closures)].
* `mut`: denotes mutability in pointer types and pattern bindings.  See [Mutability].
* `pub`: denotes public visibility in `struct` fields, `impl` blocks, and modules.  See [Crates and Modules (Exporting a Public Interface)].
* `ref`: by-reference binding.  See [Patterns (`ref` and `ref mut`)].
* `return`: return from function.  See [Functions (Early Returns)].
* `Self`: implementor type alias.  See [Traits].
* `self`: method subject.  See [Method Syntax (Method Calls)].
* `static`: global variable.  See [`const` and `static` (`static`)].
* `struct`: structure definition.  See [Structs].
* `trait`: trait definition.  See [Traits].
* `true`: boolean true literal.  See [Primitive Types (Booleans)].
* `type`: type alias, and associated type definition.  See [`type` Aliases], [Associated Types].
* `unsafe`: denotes unsafe code, functions, traits, and implementations.  See [Unsafe].
* `use`: import symbols into scope.  See [Crates and Modules (Importing Modules with `use`)].
* `where`: type constraint clauses.  See [Traits (`where` clause)].
* `while`: conditional loop.  See [Loops (`while`)].

## Operators and Symbols

* `!` (`expr!(…)`, `expr!{…}`, `expr![…]`): denotes macro expansion.  See [Macros].
* `!` (`!expr`): bitwise or logical complement.  Overloadable (`Not`).
* `%` (`expr % expr`): arithmetic remainder.  Overloadable (`Rem`).
* `%=` (`var %= expr`): arithmetic remainder & assignment.
* `&` (`expr & expr`): bitwise and.  Overloadable (`BitAnd`).
* `&` (`&expr`): borrow.  See [References and Borrowing].
* `&` (`&type`, `&mut type`, `&'a type`, `&'a mut type`): borrowed pointer type.  See [References and Borrowing].
* `&=` (`var &= expr`): bitwise and & assignment.
* `&&` (`expr && expr`): logical and.
* `*` (`expr * expr`): arithmetic multiplication.  Overloadable (`Mul`).
* `*` (`*expr`): dereference.
* `*` (`*const type`, `*mut type`): raw pointer.  See [Raw Pointers].
* `*=` (`var *= expr`): arithmetic multiplication & assignment.
* `+` (`expr + expr`): arithmetic addition.  Overloadable (`Add`).
* `+` (`trait + trait`, `'a + trait`): compound type constraint.  See [Traits (Multiple Trait Bounds)].
* `+=` (`var += expr`): arithmetic addition & assignment.
* `,`: argument and element separator.  See [Attributes], [Functions], [Structs], [Generics], [Match], [Closures], [Crates and Modules (Importing Modules with `use`)].
* `-` (`expr - expr`): arithmetic subtraction.  Overloadable (`Sub`).
* `-` (`- expr`): arithmetic negation.  Overloadable (`Neg`).
* `-=` (`var -= expr`): arithmetic subtraction & assignment.
* `->` (`fn(…) -> type`, `|…| -> type`): function and closure return type.  See [Functions], [Closures].
* `.` (`expr.ident`): member access.  See [Structs], [Method Syntax].
* `..` (`..`, `expr..`, `..expr`, `expr..expr`): right-exclusive range literal.
* `..` (`..expr`): struct literal update syntax.  See [Structs (Update syntax)].
* `..` (`variant(x, ..)`, `struct_type { x, .. }`): "and the rest" pattern binding.  See [Patterns (Ignoring bindings)].
* `...` (`expr ... expr`): inclusive range pattern.  See [Patterns (Ranges)].
* `/` (`expr / expr`): arithmetic division.  Overloadable (`Div`).
* `/=` (`var /= expr`): arithmetic division & assignment.
* `:` (`pat: type`, `ident: type`): constraints.  See [Variable Bindings], [Functions], [Structs], [Traits].
* `:` (`ident: expr`): struct field initializer.  See [Structs].
* `:` (`'a: loop {…}`): loop label.  See [Loops (Loops Labels)].
* `;`: statement and item terminator.
* `;` (`[…; len]`): part of fixed-size array syntax.  See [Primitive Types (Arrays)].
* `<<` (`expr << expr`): left-shift.  Overloadable (`Shl`).
* `<<=` (`var <<= expr`): left-shift & assignment.
* `<` (`expr < expr`): less-than comparison.  Overloadable (`Cmp`, `PartialCmp`).
* `<=` (`var <= expr`): less-than or equal-to comparison.  Overloadable (`Cmp`, `PartialCmp`).
* `=` (`var = expr`, `ident = type`): assignment/equivalence.  See [Variable Bindings], [`type` Aliases], generic parameter defaults.
* `==` (`var == expr`): comparison.  Overloadable (`Eq`, `PartialEq`).
* `=>` (`pat => expr`): part of match arm syntax.  See [Match].
* `>` (`expr > expr`): greater-than comparison.  Overloadable (`Cmp`, `PartialCmp`).
* `>=` (`var >= expr`): greater-than or equal-to comparison.  Overloadable (`Cmp`, `PartialCmp`).
* `>>` (`expr >> expr`): right-shift.  Overloadable (`Shr`).
* `>>=` (`var >>= expr`): right-shift & assignment.
* `@` (`ident @ pat`): pattern binding.  See [Patterns (Bindings)].
* `^` (`expr ^ expr`): bitwise exclusive or.  Overloadable (`BitXor`).
* `^=` (`var ^= expr`): bitwise exclusive or & assignment.
* `|` (`expr | expr`): bitwise or.  Overloadable (`BitOr`).
* `|` (`pat | pat`): pattern alternatives.  See [Patterns (Multiple patterns)].
* `|=` (`var |= expr`): bitwise or & assignment.
* `||` (`expr || expr`): logical or.
* `_`: "ignored" pattern binding.  See [Patterns (Ignoring bindings)].

## Other Syntax

<!-- Various bits of standalone stuff. -->

* `'ident`: named lifetime or loop label.  See [Lifetimes], [Loops (Loops Labels)].
* `…u8`, `…i32`, `…f64`, `…usize`, …: numeric literal of specific type.
* `"…"`: string literal.  See [Strings].
* `r"…"`, `r#"…"#`, `r##"…"##`, …: raw string literal, escape characters are not processed. See [Reference (Raw String Literals)].
* `b"…"`: byte string literal, constructs a `[u8]` instead of a string. See [Reference (Byte String Literals)].
* `br"…"`, `br#"…"#`, `br##"…"##`, …: raw byte string literal, combination of raw and byte string literal. See [Reference (Raw Byte String Literals)].
* `'…'`: character literal.  See [Primitive Types (`char`)].
* `b'…'`: ASCII byte literal.

<!-- Path-related syntax -->

* `ident::ident`: path.  See [Crates and Modules (Defining Modules)].
* `::path`: path relative to the crate root (*i.e.* an explicitly absolute path).  See [Crates and Modules (Re-exporting with `pub use`)].
* `self::path`: path relative to the current module (*i.e.* an explicitly relative path).  See [Crates and Modules (Re-exporting with `pub use`)].
* `super::path`: path relative to the parent of the current module.  See [Crates and Modules (Re-exporting with `pub use`)].
* `type::ident`: associated constants, functions, and types.  See [Associated Types].
* `<type>::…`: associated item for a type which cannot be directly named (*e.g.* `<&T>::…`, `<[T]>::…`, *etc.*).  See [Associated Types].

<!-- Generics -->

* `path<…>` (*e.g.* `Vec<u8>`): specifies parameters to generic type *in a type*.  See [Generics].
* `path::<…>`, `method::<…>` (*e.g.* `"42".parse::<i32>()`): specifies parameters to generic type, function, or method *in an expression*.
* `fn ident<…> …`: define generic function.  See [Generics].
* `struct ident<…> …`: define generic structure.  See [Generics].
* `enum ident<…> …`: define generic enumeration.  See [Generics].
* `impl<…> …`: define generic implementation.
* `for<…> type`: higher-ranked lifetime bounds.
* `type<ident=type>` (*e.g.* `Iterator<Item=T>`): a generic type where one or more associated types have specific assignments.  See [Associated Types].

<!-- Constraints -->

* `T: U`: generic parameter `T` constrained to types that implement `U`.  See [Traits].
* `T: 'a`: generic type `T` must outlive lifetime `'a`.
* `'b: 'a`: generic lifetime `'b` must outlive lifetime `'a`.
* `T: ?Sized`: allow generic type parameter to be a dynamically-sized type.  See [Unsized Types (`?Sized`)].
* `'a + trait`, `trait + trait`: compound type constraint.  See [Traits (Multiple Trait Bounds)].

<!-- Macros and attributes -->

* `#[meta]`: outer attribute.  See [Attributes].
* `#![meta]`: inner attribute.  See [Attributes].
* `$ident`: macro substitution.  See [Macros].
* `$ident:kind`: macro capture.  See [Macros].
* `$(…)…`: macro repetition.  See [Macros].

<!-- Comments -->

* `//`: line comment.  See [Comments].
* `//!`: inner line doc comment.  See [Comments].
* `///`: outer line doc comment.  See [Comments].
* `/*…*/`: block comment.  See [Comments].
* `/*!…*/`: inner block doc comment.  See [Comments].
* `/**…*/`: outer block doc comment.  See [Comments].

<!-- Various things involving parens and tuples -->

* `()`: empty tuple (*a.k.a.* unit), both literal and type.
* `(expr)`: parenthesized expression.
* `(expr,)`: single-element tuple expression.  See [Primitive Types (Tuples)].
* `(type,)`: single-element tuple type.  See [Primitive Types (Tuples)].
* `(expr, …)`: tuple expression.  See [Primitive Types (Tuples)].
* `(type, …)`: tuple type.  See [Primitive Types (Tuples)].
* `expr(expr, …)`: function call expression.  Also used to initialize tuple `struct`s and tuple `enum` variants.  See [Functions].
* `ident!(…)`, `ident!{…}`, `ident![…]`: macro invocation.  See [Macros].
* `expr.0`, `expr.1`, …: tuple indexing.  See [Primitive Types (Tuple Indexing)].

<!-- Bracey things -->

* `{…}`: block expression.
* `Type {…}`: `struct` literal.  See [Structs].

<!-- Brackety things -->

* `[…]`: array literal.  See [Primitive Types (Arrays)].
* `[expr; len]`: array literal containing `len` copies of `expr`.  See [Primitive Types (Arrays)].
* `[type; len]`: array type containing `len` instances of `type`.  See [Primitive Types (Arrays)].

[`const` and `static` (`static`)]: const-and-static.html#static
[`const` and `static`]: const-and-static.html
[`if let`]: if-let.html
[`if`]: if.html
[`type` Aliases]: type-aliases.html
[Associated Types]: associated-types.html
[Attributes]: attributes.html
[Casting Between Types (`as`)]: casting-between-types.html#as
[Closures (`move` closures)]: closures.html#move-closures
[Closures]: closures.html
[Comments]: comments.html
[Crates and Modules (Defining Modules)]: crates-and-modules.html#defining-modules
[Crates and Modules (Exporting a Public Interface)]: crates-and-modules.html#exporting-a-public-interface
[Crates and Modules (Importing External Crates)]: crates-and-modules.html#importing-external-crates
[Crates and Modules (Importing Modules with `use`)]: crates-and-modules.html#importing-modules-with-use
[Crates and Modules (Re-exporting with `pub use`)]: crates-and-modules.html#re-exporting-with-pub-use
[Enums]: enums.html
[Foreign Function Interface]: ffi.html
[Functions (Early Returns)]: functions.html#early-returns
[Functions]: functions.html
[Generics]: generics.html
[Lifetimes]: lifetimes.html
[Loops (`for`)]: loops.html#for
[Loops (`loop`)]: loops.html#loop
[Loops (`while`)]: loops.html#while
[Loops (Ending Iteration Early)]: loops.html#ending-iteration-early
[Loops (Loops Labels)]: loops.html#loop-labels
[Macros]: macros.html
[Match]: match.html
[Method Syntax (Method Calls)]: method-syntax.html#method-calls
[Method Syntax]: method-syntax.html
[Mutability]: mutability.html
[Operators and Overloading]: operators-and-overloading.html
[Patterns (`ref` and `ref mut`)]: patterns.html#ref-and-ref-mut
[Patterns (Bindings)]: patterns.html#bindings
[Patterns (Ignoring bindings)]: patterns.html#ignoring-bindings
[Patterns (Multiple patterns)]: patterns.html#multiple-patterns
[Patterns (Ranges)]: patterns.html#ranges
[Primitive Types (`char`)]: primitive-types.html#char
[Primitive Types (Arrays)]: primitive-types.html#arrays
[Primitive Types (Booleans)]: primitive-types.html#booleans
[Primitive Types (Tuple Indexing)]: primitive-types.html#tuple-indexing
[Primitive Types (Tuples)]: primitive-types.html#tuples
[Raw Pointers]: raw-pointers.html
[Reference (Byte String Literals)]: ../reference.html#byte-string-literals
[Reference (Raw Byte String Literals)]: ../reference.html#raw-byte-string-literals
[Reference (Raw String Literals)]: ../reference.html#raw-string-literals
[References and Borrowing]: references-and-borrowing.html
[Strings]: strings.html
[Structs (Update syntax)]: structs.html#update-syntax
[Structs]: structs.html
[Traits (`where` clause)]: traits.html#where-clause
[Traits (Multiple Trait Bounds)]: traits.html#multiple-trait-bounds
[Traits]: traits.html
[Unsafe]: unsafe.html
[Unsized Types (`?Sized`)]: unsized-types.html#?sized
[Variable Bindings]: variable-bindings.html
