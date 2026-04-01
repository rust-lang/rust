# stdarch-gen-arm generator guide
## Running the generator
- Run: `cargo run --bin=stdarch-gen-arm -- crates/stdarch-gen-arm/spec`
```
$ cargo run --bin=stdarch-gen-arm -- crates/stdarch-gen-arm/spec
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.18s
     Running `target/debug/stdarch-gen-arm crates/stdarch-gen-arm/spec`
```
## Input/Output
### Input files (intrinsic YAML definitions)
 - `crates/stdarch-gen-arm/spec/<feature>/*.spec.yml`
### Output files
 - Generated intrinsics:
	 - `crates/core_arch/src/<arch>/<feature>/generated.rs`
 - Generated load/store tests:
	 - `crates/core_arch/src/<arch>/<feature>/ld_st_tests_<arch>.rs`
	 - Only generated when `test: { load: <idx> }` or `test: { store: <idx> }` is set for SVE/SVE2 intrinsics.
## `.spec.yml` file anatomy
```
---
Configs
---
Variable definitions
---

Intrinsic definitions

---
```
- If you're new to YAML syntax, consider [reviewing](https://quickref.me/yaml.html) some of the less obvious syntax and features.
- For example, mapping an attribute to a sequence can be done in two different ways:
```yaml
attribute: [item_a, item_b, item_c]
```
or
```yaml
attribute:
    - item_a
    - item_b
    - item_c
``` 
## Configs
- Mappings defining top-level settings applied to all intrinsics:
- `arch_cfgs`
    - Sequence of mappings specifying `arch_name`, `target_feature` (sequence), and `llvm_prefix`.
- `uses_neon_types`(_Optional_)
    - A boolean specifying whether to emit NEON type imports in generated code.
- `auto_big_endian`(_Optional_)
    - A boolean specifying whether to auto-generate big-endian shuffles when possible.
- `auto_llvm_sign_conversion`(_Optional_)
    - A boolean specifying whether to auto-convert LLVM wrapper args to signed types.
## Variable definitions
- Defines YAML anchors/variables to avoid repetition.
- Commonly used for stability attributes, cfgs and target features.
## Intrinsic definitions
### Example
```yaml
    - name: "vtst{neon_type[0].no}"
      doc: "Signed compare bitwise Test bits nonzero"
      arguments: ["a: {neon_type[0]}", "b: {neon_type[0]}"]
      return_type: "{neon_type[1]}"
      attr:
        - FnCall: [cfg_attr, [test, {FnCall: [assert_instr, [cmtst]]}]]
        - FnCall: [stable, ['feature = "neon_intrinsics"', 'since = "1.59.0"']]
      safety: safe
      types:
        - [int64x1_t, uint64x1_t, 'i64x1', 'i64x1::new(0)']
        - [int64x2_t, uint64x2_t, 'i64x2', 'i64x2::new(0, 0)']
        - [poly64x1_t, uint64x1_t, 'i64x1', 'i64x1::new(0)']
        - [poly64x2_t, uint64x2_t, 'i64x2', 'i64x2::new(0, 0)']
      compose:
        - Let: [c, "{neon_type[0]}", {FnCall: [simd_and, [a, b]]}]
        - Let: [d, "{type[2]}", "{type[3]}"]
        - FnCall: [simd_ne, [c, {FnCall: [transmute, [d]]}]]
```

### Explanation of fields
- `name`
    - The name of the intrinsic
    - Often built from a base name followed by a type suffix
- `doc` (_Optional_)
    - A string explaining the purpose of the intrinsic
- `static_defs` (_Optional_)
    - A sequence of const generics of the format `"const <NAME>: <type>"`
- `arguments`
    - A sequence of strings in the format `"<argname>: <argtype>"`
- `return_type` (_Optional_)
    - A string specifying the return type. If omitted, the intrinsic returns `()`.
- `attr` (_Optional_)
    - A sequence of items defining the attributes to be applied to the intrinsic. Often stability attributes, target features, or `assert_instr` tests. At least one of `attr` or `assert_instr` must be set.
- `target_features` (_Optional_)
    - A sequence of target features to enable for this intrinsic (merged with any global `arch_cfgs` settings).
- `assert_instr` (_Optional_)
    - A sequence of strings expected to be found in the assembly. Required if `attr` is not set.
- `safety` (_Optional_)
    - Use `safe`, or map `unsafe:` to a sequence of unsafety comments:
        - `custom: "<string>"`
        - `uninitialized`
        - `pointer_offset`, `pointer_offset_vnum`, or `dereference` (optionally qualified with `predicated`, `predicated_non_faulting`, or `predicated_first_faulting`)
        - `unpredictable_on_fault`
        - `non_temporal`
        - `neon`
        - `no_provenance: "<string>"`
- `substitutions` (_Optional_)
    - Mappings of custom wildcard names to either `MatchSize` or `MatchKind` expressions
- `types`
    - A sequence or sequence of sequences specifying the types to use when producing each intrinsic variant. These sequences can then be indexed by wildcards.
- `constraints` (_Optional_)
    - A sequence of mappings. Each specifies a variable and a constraint. The available mappings are:
    - Assert a variable's value exists in a sequence of i32's
        - Usage: `{ variable: <name>, any_values: [<i32>,...] }`
    - Assert a variable's value exists in a range (inclusive)
        - Usage: `{ variable: <name>, range: [<i32>, <i32>] }`
    - Assert a variable's value exists in a range via a match (inclusive)
        - Usage: `{ variable: <name>, range: <MatchSize returning [i32,i32]> }`
    - Assert a variable's value does not exceed the number of elements in a SVE type `<type>`.
        - Usage: `{ variable: <name>, sve_max_elems_type: <type> }`
    - Assert a variable's value does not exceed the number of elements in a vector type `<type>`.
        - Usage: `{ variable: <name>, vec_max_elems_type: <type> }`
- `predication_methods` (_Optional_)
    - Configuration for predicate-form variants. Only used when the intrinsic name includes an `_m*_` wildcard (e.g., `{_mx}`, `{_mxz}`).
    - `zeroing_method`: Required when requesting `_z`; either `{ drop: <arg> }` to remove an argument and replace it with a zero initialiser, or `{ select: <predicate_var> }` to select zeros into a predicate.
    - `dont_care_method`: How `_x` should be implemented (`inferred`, `as_zeroing`, or `as_merging`).
- `compose`
    - A sequence of expressions that make up the body of the intrinsic
- `big_endian_inverse` (_Optional_)
    - A boolean, default false. If true, generates two implementations of each intrinsic variant, one for each endianness, and attempts to automatically generate the required bit swizzles
- `visibility` (_Optional_)
    - Function visibility. One of `public` (default) or `private`.
- `n_variant_op` (_Optional_)
    - Enables generation of an `_n` variant when the intrinsic name includes the `{_n}` wildcard. Set to the operand name that should be splattered for the `_n` form.
- `test` (_Optional_)
	- When set, load/store tests are automatically generated.
    - A mapping of either `load` or `store` to a number that indexes `types` to specify the type that the test should be addressing in memory. 
### Expressions
#### Common
- `Let`
    - Defines a variable
    - Usage: `Let: [<variable>, <type(optional)>, <expression>]`
- `Const`
    - Defines a const
    - Usage: `Const: [<variable>, <type>, <expression>]`
- `Assign`
    - Performs variable assignment
    - Usage: `Assign: [<variable>, <expression>]`
- `FnCall`
    - Performs a function call
    - Usage: `FnCall: [<function pointer: expression>, [<argument: expression>, ... ], [<turbofish argument: expression>, ...](optional), <unsafe wrapper(optional): bool>]`
- `MacroCall`
    - Performs a macro call
    - Usage: `MacroCall: [<macro name>, <token stream>]`
- `MethodCall`
    - Performs a method call
    - Usage: `MethodCall: [<object: expression>, <method name>, [<argument: expression>, ... ]]`
- `LLVMLink`
    - Creates an LLVM link and stores the function's name in the wildcard `{llvm_link}` for later use in subsequent expressions.
    - If left unset, the arguments and return type inherit from the intrinsic's signature by default. The links will also be set automatically if unset.
    - Usage:
```yaml
LLVMLink:
    name: <name>
    arguments: [<expression>, ... ](optional)
    return_type: <return type>(optional)
    links: (optional)
        - link: <link>
          arch: <arch>
        - ...
```
- `Identifier`
    - Emits a symbol. Prepend with a `$` to treat it as a scope variable, which engages variable tracking and enables inference. For example, `my_function_name` for a generic symbol or `$my_variable` for a variable.
    - Usage `Identifier: [<symbol name>, <Variable|Symbol>]`
- `CastAs`
    - Casts an expression to an unchecked type
    - Usage: `CastAs: [<expression>, <type>]`
- `MatchSize`
    - Allows for conditional generation depending on the size of a specified type
    - Usage:
```yaml
MatchSize:
    - <type>
    - default: <expression>
      byte(optional): <expression>
      halfword(optional): <expression>
      doubleword(optional): <expression>
```
- `MatchKind`
    - Allows for conditional generation depending on the kind of a specified type
```yaml
MatchKind:
    - <type>
    - default: <expression>
      float(optional): <expression>
      unsigned(optional): <expression>
```
#### Rarely Used
- `IntConstant`
    - Constant signed integer expression
    - Usage: `IntConstant: <i32>`
- `FloatConstant`
    - Constant floating-point expression
    - Usage: `FloatConstant: <f32>`
- `BoolConstant`
    - Constant boolean expression
    - Usage: `BoolConstant: <bool>`
- `Array`
    - An array of expressions
    - Usage: `Array: [<expression>, ...]`
- `SvUndef`
    - Returns the LLVM `undef` symbol
    - Usage: `SvUndef`
- `Multiply`
    - Simply `*`
    - Usage: `Multiply: [<expression>, <expression>]`
- `Xor`
    - Simply `^`
    - Usage: `Xor: [<expression>, <expression>]`
- `ConvertConst`
    - Converts the specified constant to the specified type's kind
    - Usage: `ConvertConst: [<type>, <i32>]`
- `Type`
    - Yields the given type in the Rust representation
    - Usage: `Type: [<type>]`

### Wildstrings
- Wildstrings let you take advantage of wildcards.
- For example, they are often used in intrinsic names `name: "vtst{neon_type[0].no}"`
- As shown above, wildcards are identified by the surrounding curly brackets.
- Double curly brackets can be used to escape wildcard functionality if you need literal curly brackets in the generated intrinsic.
### Wildcards
Wildcards are heavily used in the spec. They let you write generalised definitions for a group of intrinsics that generate multiple variants. The wildcard itself is replaced with the relevant string in each variant.
Ignoring endianness, for each row in the `types` field of an intrinsic in the spec, a variant of the intrinsic will be generated. That row's contents can be indexed by the wildcards. Below is the behaviour of each wildcard.
- `type[<index: usize>]`
    - Replaced in each variant with the value in the indexed position in the relevant row of the `types` field.
    - For unnested sequences of `types` (i.e., `types` is a sequence where each element is a single item, not another sequence), the square brackets can be omitted. Simply: `type`
- `neon_type[<index: usize>]`
    - Extends the behaviour of `type` with some NEON-specific features and inference.
    - Tuples: This wildcard can also be written as `neon_type_x<n>` where `n` is in the set `{2,3,4}`. This generates the `n`-tuple variant of the (inferred) NEON type.
    - Suffixes: These modify the behaviour of the wildcard from simple substitution.
	    - `no` -  normal behaviour. Tries to do as much work as it can for you, inferring when to emit:
            - Regular type-size suffixes: `_s8`, `_u16`, `_f32`, ...
            - `q` variants for double-width (128b) vector types: `q_s8`, `q_u16`, `q_f32`, ...
            - `_x<n>` variants for tuple vector types: `_s8_x2`, `_u32_x3`, `_f64_x4`, ...
            - As well as any combination of the above: `q_s16_x16` ...
    - Most of the other suffixes modify the normal behaviour by disabling features or adding new ones. (See table below)
- `sve_type[<index: usize>]`
    - Similar to `neon_type`, but without the suffixes.
- `size[<index: usize>]`
    - The size (in bits) of the indexed type.
- `size_minus_one[<index: usize>]`
    - Emits the size (in bits) of the indexed type minus one.
- `size_literal[<index: usize>]`
    - The literal representation of the indexed type.
    - `b`: byte, `h`: halfword, `w`: word, or `d`: double.
- `type_kind[<index: usize>]`
    - The literal representation of the indexed type's kind.
    - `f`: float, `s`: signed, `u`: unsigned, `p`: polynomial, `b`: boolean.
- `size_in_bytes_log2[<index: usize>]`
    - Log2 of the size of the indexed type in *bytes*.
- `predicate[<index: usize>]`
    - SVE predicate vector type inferred from the indexed type.
- `max_predicate`
    - The same as predicate, but uses the largest type in the relevant `types` sequence/row.
- `_n`
    - Emits the current N-variant suffix when `n_variant_op` is configured.
- `<wildcard> as <type>`
    - If `<wildcard>` evaluates to a vector, it produces a vector of the same shape, but with `<type>` as the base type.
- `llvm_link`
    - If the `LLVMLink` mapping has been set for an intrinsic, this will give the name of the link.
- `_m*`
    - Predicate form masks. Use wildcards such as `{_mx}` or `{_mxz}` to expand merging/don't-care/zeroing variants according to the mask.
- `<custom>`
    - You may simply call upon wildcards defined under `substitutions`.
### neon_type suffixes

| suffix            | implication                                   |
| ----------------- | --------------------------------------------- |
| `.no`             | Normal                                        |
| `.noq`            | Never include `q`s                            |
| `.nox`            | Never include `_x<n>`s                        |
| `.N`              | Include `_n_`, e.g., `_n_s8`                  |
| `.noq_N`          | Include `_n_`, but never `q`s                 |
| `.dup`            | Include `_dup_`, e.g., `_dup_s8`              |
| `.dup_nox`        | Include `_dup_` but never `_x<n>`s            |
| `.lane`           | Include `_lane_`, e.g., `_lane_s8`            |
| `.lane_nox`       | Include `_lane_`, but never `_x<n>`s          |
| `.rot90`          | Include `_rot90_`, e.g., `_rot90_s8`          |
| `.rot180`         | Include `_rot180_`, e.g., `_rot180_s8`        |
| `.rot270`         | Include `_rot270_`, e.g., `_rot270_s8`        |
| `.rot90_lane`     | Include `_rot90_lane_`                        |
| `.rot180_lane`    | Include `_rot180_lane_`                       |
| `.rot270_lane`    | Include `_rot270_lane_`                       |
| `.rot90_laneq`    | Include `_rot90_laneq_`                       |
| `.rot180_laneq`   | Include `_rot180_laneq_`                      |
| `.rot270_laneq`   | Include `_rot270_laneq_`                      |
| `.base`           | Produce only the size, e.g., `8`, `16`        |
| `.u`              | Produce the type's unsigned equivalent        |
| `.laneq_nox`      | Include `_laneq_`, but never `_x<n>`s         |
| `.tuple`          | Produce only the size of the tuple, e.g., `3` |
| `.base_byte_size` | Produce only the size in bytes.               |
 
