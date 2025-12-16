# LLDB - Python Providers

> NOTE: LLDB's C++<->Python FFI expects a version of python designated at the time LLDB was
>compiled. LLDB is careful to correspond this version to the minimum in typical Linux and macOS
>distributions, but on Windows there is no easy solution. If you recieve an import error regarding
>`_lldb` not existing, a mismatched Python version is likely the cause.
>
> LLDB is considering solutions this issue. For updates, see
>[this discussion][minimal_python_install] and [this github issue][issue_167001]

[minimal_python_install]: https://discourse.llvm.org/t/a-minimal-python-install-for-lldb/88658
[issue_167001]: https://github.com/llvm/llvm-project/issues/167001

> NOTE: Currently (Nov 2025), LLDB's minimum supported Python version is 3.8 with plans to update it to
>3.9 or 3.10 depending on several outside factors. Scripts should ideally be written with only the
>features available in the minimum supported Python version. Please see [this discussion][mrpv] for
>more info.

[mrpv]: https://discourse.llvm.org/t/rfc-upgrading-llvm-s-minimum-required-python-version/88605/

> NOTE: The path to LLDB's python package can be located via the CLI command `lldb -P`

LLDB provides 3 mechanisms for customizing output:

* Formats
* Synthetic providers
* Summary providers

## Formats

The official documentation is [here](https://lldb.llvm.org/use/variable.html#type-format). In short,
formats allow one to set the default print format for primitive types (e.g. print `25u8` as decimal
`25`, hex `0x19`, or binary `00011001`).

Rust will almost always need to override `unsigned char`, `signed char`, `char`, `u8`, and `i8`, to
(unsigned) decimal format.

## Synthetic Providers

The official documentation is [here](https://lldb.llvm.org/use/variable.html#synthetic-children),
but some information is vague, outdated, or entirely missing.

Nearly all interaction the user has with variables will be through LLDB's
[`SBValue` objects][sbvalue] which are used both in the Python API, and internally via LLDB's
plugins and CLI.

[sbvalue]: https://lldb.llvm.org/python_api/lldb.SBValue.html

A Synthetic Provider is a Python class, written with a specific interface, that is associated with
one or more Rust types. The Synthetic Provider wraps `SBValue` objects and LLDB will call our
class's functions when inspecting the variable.

The wrapped value is still an `SBValue`, but when calling e.g. `SBValue.GetChildAtIndex`, it will
internally call `SyntheticProvider.get_child_at_index`. You can check if a value has a synthetic
provider via `SBValue.IsSynthetic()`, and which synthetic it is via `SBValue.GetTypeSynthetic()`. If
you want to interact with the underlying non-synthetic value, you can call
`SBValue.GetNonSyntheticValue()`.


The expected interface is as follows:

```python
class SyntheticProvider:
    def __init__(self, valobj: SBValue, _lldb_internal): ...

    # optional
    def update(self) -> bool: ...

    # optional
    def has_children(self) -> bool: ...

    # optional
    def num_children(self, max_children: int) -> int: ...

    def get_child_index(self, name: str) -> int: ...

    def get_child_at_index(self, index: int) -> SBValue: ...

    # optional
    def get_type_name(self) -> str: ...

    # optional
    def get_value(self) -> SBValue: ...
```

Below are explanations of the methods, their quirks, and how they should generally be used. If a
method overrides an `SBValue` method, that method will be listed.

### `__init__`

This function is called once per object, and must store the `valobj` in the python class so that it
is accessible elsewhere. Very little else should be done here.

### (optional) `update`

This function is called prior to LLDB interacting with a variable, but after `__init__`. LLDB tracks
whether `update` has already been called. If it has been, and if it is not possible for the variable
to have changed (e.g. inspecting the same variable a second time without stepping), it will omit the
call to `update`.

This function has 2 purposes:

* Store/update any information that may have changed since the last time `update` was run
* Inform LLDB if there were changes to the children such that it should flush the child cache.

Typical operations include storing the heap pointer, length, capacity, and element type of a `Vec`,
determining an enum variable's variant, or checking which slots of a `HashMap` are occupied.

The bool returned from this function is somewhat complicated, see:
[`update` caching](#update-caching) below for more info. When in doubt, return `False`/`None`.
Currently (Nov 2025), none of the visualizers return `True`, but that may change as the debug info
test suite is improved.

#### `update` caching

LLDB attempts to cache values when possible, including child values. This cache is effectively the
number of child objects, and the addresses of the underlying debugee memory that the child object
represents. By returning `True`, you indicate to LLDB that the number of children and the addresses
of those children have not changed since the last time `update` was run, meaning it can reuse the
cached children.

**Returning `True` in the wrong circumstances will result in the debugger outputting incorrect
information**.

Returning `False` indicates that there have been changes, the cache will be flushed, and the
children will be fetched from scratch. It is the safer option if you are unsure.

The only relationship that matters is parent-to-child. Grandchildren depend on the `update` function
of their direct parent, not that of the grandparent.

It is important to view the child cache as pointers-to-memory. For example, if a slice's `data_ptr`
value and `length` have not changed, returning `True` is appropriate. Even if the slice is mutable
and elements of it are overwritten (e.g. `slice[0] = 15`), because the child cache consists of
*pointers*, they will reflect the new data at that memory location.

Conversely, if `data_ptr` has changed, that means it is pointing to a new location in memory, the
child pointers are invalid, and the cache must be flushed. If the `length` has changed, we need to
flush the cache to reflect the new number of children. If `length` has changed but `data_ptr` has
not, it is possible to store the old children in the `SyntheticProvider` itself (e.g.
`list[SBValue]`) and dole those out rather than generating them from scratch, only creating new
children if they do not already exist in the `SyntheticProvider`'s list.

For further clarification, see [this discussion](https://discourse.llvm.org/t/when-is-it-safe-to-cache-syntheticprovider-update/88608)

> NOTE: when testing the caching behavior, do not rely on LLDB's heuristic to persist variables when
> stepping. Instead, store the variable in a python object (e.g. `v = lldb.frame.var("var_name")`),
> step forward, and then inspect the stored variable.

### (optional) `has_children`

> Overrides `SBValue.MightHaveChildren`

This is a shortcut used by LLDB to check if the value has children *at all*, without doing
potentially expensive computations to determine how many children there are (e.g. linked list).
Often, this will be a one-liner of `return True`/`return False` or
`return self.valobj.MightHaveChildren()`.

### (optional) `num_children`

> Overrides `SBValue.GetNumChildren`

Returns the total number of children that LLDB should try to access when printing the type. This
number **does not** need to match to total number of synthetic children.

The `max_children` argument can be returned if calculating the number of children can be expensive
(e.g. linked list). If this is not a consideration, `max_children` can be omitted from the function
signature.

Additionally, fields can be intentionally "hidden" from LLDB while still being accessible to the
user. For example, one might want a `vec![1, 2, 3]` to display only its elements, but still have the
`len` and `capacity` values accessible on request. By returning `3` from `num_children`, one can
restrict LLDB to only displaying `[1, 2, 3]`, while users can still directly access `v.len` and
`v.capacity`. See: [Example Provider: Vec\<T\>](#example-provider-vect) to see an implementation of
this.

### `get_child_index`

> Overrides `SBValue.GetIndexOfChildWithName`
>
> Affects `SBValue.GetChildMemberWithName`

Given a name, returns the index that the child should be accessed at. It is expected that the return
value of this function is passed directly to `get_child_at_index`. As with `num_children`, the
values returned here *can* be arbitrary, so long as they are properly coordinated with
`get_child_at_index`.

One special value is `$$dereference$$`. Accounting for this pseudo-field will allow LLDB to use the
`SBValue` returned from `get_child_at_index` as the result of a dereference via LLDB's expression
parser (e.g. `*val` and `val->field`)

### `get_child_at_index`

> Overrides `SBValue.GetChildAtIndex`

Given an index, returns a child `SBValue`. Often these are generated via
`SBValue.CreateValueFromAddress`, but less commonly `SBValue.CreateChildAtOffset`,
`SBValue.CreateValueFromExpression`, and `SBValue.CreateValueFromData`. These functions can be a
little finicky, so you may need to fiddle with them to get the output you want.

In some cases, `SBValue.Clone` is appropriate. It creates a new child that is an exact copy of an
existing child, but with a new name. This is useful for cases like tuples, which have field names of
the style `__0`, `__1`, ... when we would prefer they were named `0`, `1`, ...

Small alterations can be made to the resulting child before it is returned. This is useful for
`&str`/`String`, where we would prefer if the children were displayed as
`lldb.eFormatBytesWithASCII` rather than just as a decimal value.

### (optional) `get_type_name`

> Overrides `SBValue.GetDisplayTypeName`

Overrides the displayed name of a type. For a synthetic `SBValue` whose type name is overridden, the
original type name can still be retrieved via `SBValue.GetTypeName()` and
`SBValue.GetType().GetName()`

This can be helpful in shortening the name of common standard library types (e.g.
`std::collections::hash::map::HashMap<K, V, std::hash::random::RandomState>` -> `HashMap<K, V>`), or
in normalizing MSVC type names (e.g. `ref$<str$>` -> `&str`).

The string manipulation can be a little tricky, especially on MSVC where we cannot conveniently
access the generic parameters of the type.

### (optional) `get_value`

> Overrides `SBValue.GetValue()`, `SBValue.GetValueAsUnsigned()`, `SBValue.GetValueAsSigned()`,
>`SBValue.GetValueAsAddress()`,

The `SBValue` returned is expected to be a primitive type or pointer, and is treated as the value of
the variable in expressions.

> IMPORTANT: The `SBValue` returned **must be stored in the `SyntheticProvider`**. There is
>currently (Nov 2025) a bug where if the `SBValue` is acquired within `get_value` and not stored
>anywhere, Python will segfault when LLDB attempts to access the value.

## Summary Providers

Summary providers are python functions of the following form:

```python
def SummaryProvider(valobj: SBValue, _lldb_internal) -> str: ...
```

Where the returned string is passed verbatim to the user. If the returned value isn't a string, it
is naively convered to a string (e.g. `return None` prints `"None"`, not an empty string).

If the `SBValue` passed in is of a type that has a Synthetic Provider, `valobj.IsSynthetic()` will
return `True`, and the synthetic's corresponding functions will be used. If this is undesirable, the
original value can be retrieved via `valobj.GetNonSyntheticValue()`. This can be helpful in cases
like `String`, where individually calling `GetChildAtIndex` in a loop is much slower than accessing
the heap pointer, reading the whole byte array directly from the debugee's memory, and using
Python's `bytes.decode()`.

### Instance Summaries

Regular `SummaryProvider` functions take an opaque `SBValue`. That `SBValue` will reflect the type's
`SyntheticProvider` if one exists, but we cannot access the `SyntheticProvider` instance itself, or
any of its internal implementation details. This is deterimental in cases where we need some of
those internal details to help complete the summary. Currently (Nov 2025), in the synthetic we just
run the non-synthetic value through the synthetic provider
(`synth = SyntheticProvider(valobj.GetNonSyntheticValue(), _dict)`), but this is obviously
suboptimal and there are plans to use the method outlined below.

Instead, we can leverage the Python module's state to allow for instance summaries. Prior art for
this technique exists in the [old CodeLLDB Rust visualizer scripts](https://github.com/vadimcn/codelldb/blob/cf9574977b80e29c6de2c44d12f1071a53a54caf/formatters/rust.py#L110).

In short: every Synthetic Provider's `__init__` function stores a unique ID and a weak reference to
`self` in a global dictionary. The Synthetic Provider class also implements a `get_summary`
function. The type's `SummaryProvider` is a function that looks up the unique ID in this dictionary,
then calls a `get_summary` on the instance it retrieves.

```python
import weakref

SYNTH_BY_ID = weakref.WeakValueDictionary()

class SyntheticProvider:
    valobj: SBValue

    # slots requires opting-in to __weakref__
    __slots__ = ("valobj", "__weakref__")

    def __init__(valobj: SBValue, _dict):
        SYNTH_BY_ID[valobj.GetID()] = self
        self.valobj = valobj

    def get_summary(self) -> str:
        ...

def InstanceSummaryProvider(valobj: SBValue, _dict) -> str:
    # GetNonSyntheticValue should never fail as InstanceSummaryProvider implies an instance of a
    # `SyntheticProvider`. No non-synthetic types should ever have this summary assigned to them
    # We use GetNonSyntheticValue because the synthetic vaobj has its own unique ID
    return SYNTH_BY_ID[valobj.GetNonSyntheticValue().GetID()].get_summary()
```

For example, one might use this for the Enum synthetic provider. The summary would like to access
the variant name, but there isn't a convenient way to reflect this via the type name or child-values
of the synthetic. By implementing an instance summary, we can retrieve the variant name via
`self.variant.GetTypeName()` and some string manipulation.

# Writing Visualizer Scripts

> IMPORTANT: Unlike GDB and CDB, LLDB can debug executables with either DWARF or PDB debug info.
>Visualizers must be written to account for both formats whenever possible. See:
>[rust-codegen](./rust-codegen.md#dwarf-vs-pdb) for an overview of the differences

Scripts are injected into LLDB via the CLI command `command script import <path-to-script>.py`. Once
injected, classes and functions can be added to the synthetic/summary pool with `type synthetic add`
and `type summary add` respectively. The summaries and synthetics can be associated with a
"category", which is typically named after the language the providers are intended for. The category
we use will be called `Rust`.

> TIP: all LLDB commands can be prefixed with `help` (e.g. `help type synthetic add`) for a brief
description, list of arguments, and examples.

Currently (Nov 2025) we use `command source ...`, which executes a series of CLI commands from the
file [`lldb_commands`](https://github.com/rust-lang/rust/blob/main/src/etc/lldb_commands) to add
providers. This file is somewhat unwieldy, and will soon be supplanted by the Python API equivalent
outlined below.

## `__lldb_init_module`

This is an optional function of the form:

```python
def __lldb_init_module(debugger: SBDebugger, _lldb_internal) -> None: ...
```

This function is called at the end of `command script import ...`, but before control returns back
to the CLI. It allows the script to initialize its own state.

Crucially, it is passed a reference to the debugger itself. This allows us to create the `Rust`
category and add providers to it. It can also allow us to conditionally change which providers we
use depending on what version of LLDB the script detects. This is vital for backwards compatibility
once we begin using recognizer functions, as recognizers were added in lldb 19.0.

## Visualizer Resolution

The order that visualizers resolve in is listed [here][formatters_101]. In short:

[formatters_101]: https://lldb.llvm.org/use/variable.html#finding-formatters-101

* If there is an exact match (non-regex name, recognizer function, or type already matched to
provider), use that
* If the object is a pointer/reference, try to use the dereferenced type's formatter
* If the object is a typedef, check the underlying type for a formatter
* If none of the above work, iterate through the regex type matchers

Within each of those steps, **iteration is done backwards** to allow new commands to "override" old
commands. This is important for cases like `Box<str>` vs `Box<T>`, were we want a specialized
synthetic for the former, but a more generalized synthetic for the latter.

## Minutiae

LLDB's API is very powerful, but there are some "gotchas" and unintuitive behavior, some of which
will be outlined below. The python implementation can be viewed at the path returned by the CLI
command `lldb -P` in `lldb\__init__.py`. In addition to the
[examples in the lldb repo][synth_examples], there are also [C++ visualizers][plugin_cpp] that can
be used as a reference (e.g. [LibCxxVector, the equivalent to `Vec<T>`][cxx_vector]). While C++'s
visualizers are written in C++ and have access to LLDB's internals, the API and general practices
are very similar.

[synth_examples]:https://github.com/llvm/llvm-project/tree/main/lldb/examples/synthetic
[plugin_cpp]: https://github.com/llvm/llvm-project/tree/main/lldb/source/Plugins/Language/CPlusPlus
[cxx_vector]: https://github.com/llvm/llvm-project/blob/main/lldb/source/Plugins/Language/CPlusPlus/LibCxxVector.cpp

### `SBValue`

* Pointer/reference `SBValue`s will effectively "auto-deref" in some cases, acting as if the
children of the pointed-to-object are its own children.
* The non-function fields are typically [`property()`][property] fields that point directly to the
function anyway (e.g. `SBValue.type = property(GetType, None)`). Accessing through these shorthands
is a bit slower to access than just calling the function directly, so they should be avoided. Some
of the properties return special objects with special properties (e.g. `SBValue.member` returns an
object that acts like `dict[str, SBValue]` to access children). Internally, many of these special
objects just allocate a new class instance and call the function on the `SBValue` anyway, resulting
in additional performance loss (e.g. `SBValue.member` internally just implements `__getitem__` which
is the one-liner `return self.valobj.GetChildMemberWithName(name)`)
* `SBValue.GetID` returns a unique `int` for each value for the duration of the debug session.
Synthetic `SBValue`'s have a different ID than their underlying `SBValue`. The underlying ID can be
retrieved via `SBValue.GetNonSyntheticValue().GetID()`.
* When manually calculating an address, `SBValue.GetValueAsAddress` should be preferred over
`SBValue.GetValueAsUnsigned` due to [target-specific behavior][get_address]
* Getting a string representation of an `SBValue` can be tricky because `GetSummary` requires a
summary provider and `GetValue` requires the type be representable by a primitive. In almost all
cases where neither of those conditions are met, the type is a user defined struct that can be
passed through `StructSummaryProvider`.

[property]: https://docs.python.org/3/library/functions.html#property
[get_address]: https://lldb.llvm.org/python_api/lldb.SBValue.html#lldb.SBValue.GetValueAsAddress

### `SBType`

* "Aggregate type" means a non-primitive struct/class/union
* "Template" is equivalent to "Generic"
* Types can be looked up by their name via `SBTarget.FindFirstType(type_name)`. `SBTarget` can be
acquired via `SBValue.GetTarget`
* `SBType.template_args` returns `None` instead of an empty list if the type has no generics
* It is sometimes necessary to transform a type into the type you want via functions like
`SBType.GetArrayType` and `SBType.GetPointerType`. These functions cannot fail. They ask the
underlying LLDB `TypeSystem` plugin for the type, bypassing the debug info completely. Even if the
type does not exist in the debug info at all, these functions can create the appropriate type.
* `SBType.GetCanonicalType` is effectively `SBType.GetTypedefedType` + `SBType.GetUnqualifiedType`.
Unlike `SBType.GetTypedefedType`, it will always return a valid `SBType` regardless of whether or
not the original `SBType` is a typedef.
* `SBType.GetStaticFieldWithName` was added in LLDB 18. Unfortunately, backwards compatibility isn't
always possible since the static fields are otherwise completely inaccessible.


# Example Provider: `Vec<T>`

## SyntheticProvider

We start with the typical prelude, using `__slots__` since we have known fields. In addition to the
object itself, we also need to store the type of the elements because `Vec`'s heap pointer is a
`*mut u8`, not a `*mut T`. Rust is a statically typed language, so the type of `T` will never
change. That means we can store it during initialization. The heap pointer, length, and capacity
*can* change though, and thus are default initialized here.

```python
import lldb

class VecSyntheticProvider:
    valobj: SBValue
    data_ptr: SBValue
    len: int
    cap: int
    element_type: SBType

    __slots__ = (
        "valobj",
        "data_ptr",
        "len",
        "cap",
        "element_type",
        "__weakref__",
    )

    def __init__(valobj: SBValue, _dict) -> None:
        self.valobj = valobj
        # invalid type is a better default than `None`
        self.element_type = SBType()

        # special handling to account for DWARF/PDB differences
        if (arg := valobj.GetType().GetTemplateArgumentType(0)):
            self.element_type = arg
        else:
            arg_name = next(get_template_args(valobj.GetTypeName()))
            self.element_type = resolve_msvc_template_arg(arg_name, valobj.GetTarget())
```

For the implementation of `get_template_args` and `resolve_msvc_template_arg`, please see:
[`lldb_providers.py`](https://github.com/rust-lang/rust/blob/main/src/etc/lldb_providers.py#L136).

Next, the update function. We check if the pointer or length have changed. We can ommit checking the
capacity, as the number of children will remain the same unless `len` changes. If changing the
capacity resulted in a reallocation, `data_ptr`'s address would be different.

If `data_ptr` and `length` haven't changed, we can take advantage of LLDB's caching and return
early. If they have changed, we store the new values and tell LLDB to flush the cache.

```python
def update(self):
    ptr = self.valobj.GetChildMemberWithName("data_ptr")
    len = self.valobj.GetChildMemberWithName("length").GetValueAsUnsigned()

    if (
        self.data_ptr.GetValueAsAddress() == ptr.GetValueAsAddress()
        and self.len == len
    ):
        # Our child address offsets and child count are still valid
        # so we can reuse cached children
        return True

    self.data_ptr = ptr
    self.len = len

    return False
```

`has_children` and `num_children` are both straightforward:

```python
def has_children(self) -> bool:
    return True

def num_children(self) -> int:
    return self.len
```

When accessing elements, we expect values of the format `[0]`, `[1]`, etc. to mimic indexing.
Additionally, we still want the user to be able to quickly access the length and capacity, as they
can be very useful when debugging. We assign these values `u32::MAX - 1` and `u32::MAX - 2`
respectively, as we can almost surely guarantee that they will not overlap with element values. Note
that we can account for both the full and shorthand `capacity` name.

```python
    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        if name == "len":
            return lldb.UINT32_MAX - 1
        if name == "cap" or name == "capacity":
            return lldb.UINT32_MAX - 2

        return -1
```

We now have to properly coordinate `get_child_at_index` so that the elements, length, and capacity
are all accessible.

```python
def get_child_at_index(self, index: int) -> SBValue:
    if index == UINT32_MAX - 1:
        return self.valobj.GetChildMemberWithName("len")
    if index == UINT32_MAX - 2:
        return (
            self.valobj.GetChildMemberWithName("buf")
            .GetChildMemberWithName("inner")
            .GetChildMemberWithName("cap")
            .GetChildAtIndex(0)
            .Clone("capacity")
        )
    addr = self.data_ptr.GetValueAsAddress()
    addr += index * self.element_type.GetByteSize()
    return self.valobj.CreateValueFromAddress(f"[{index}]", addr, self.element_type)
```

For the type's display name, we can strip the path qualifier. User defined types named
`Vec` will end up fully qualified, so there shouldn't be any ambiguity. We can also remove the
allocator generic, as it's very very rarely useful. We use `get_template_args` instead of
`self.element_type.GetName()` for 3 reasons:

1. If we fail to resolve the element type for any reason, `self.valobj`'s type name can still let
the user know what the real type of the element is
2. Type names are not subject to the limitations of DWARF and PDB nodes, so the template type in
the name will reflect things like `*const`/`*mut` and `&`/`&mut`.
3. We do not currently (Nov 2025) normalize MSVC type names, but once we do, we will need to work with the
string-names of types anyway. It's also much easier to cache a string-to-string conversion compared
to an `SBType`-to-string conversion.

```python
def get_type_name(self) -> str:
    return f"Vec<{next(get_template_args(self.valobj))}>"
```

There isn't an appropriate primitive value with which to represent a `Vec`, so we simply ommit
the `get_value` function.

## SummaryProvider

The summary provider is very simple thanks to our synthetic provider. The only real hiccup is that
`GetSummary` only returns a value if the object's type has a `SummaryProvider`. If it doesn't, it
will return an empty string which is not ideal. In a full set of visualizer scripts, we can ensure
that every type that doesn't have a `GetSummary()` or a `GetValue()` is a struct, and then delegate
to a generic `StructSummaryProvider`. For this demonstration, I will gloss over that detail.

```python
def VecSummaryProvider(valobj: SBValue, _lldb_internal) -> str:
    children = []
    for i in range(valobj.GetNumChildren()):
        child = valobj.GetChildAtIndex(i)
        summary = child.GetSummary()
        if summary is None:
            summary = child.GetValue()
            if summary is None:
                summary = "{...}"

        children.append(summary)

    return f"vec![{", ".join(children)}]"
```

## Enabling the providers

Assume this synthetic is imported into `lldb_lookup.py`

With CLI commands:

```txt
type synthetic add -l lldb_lookup.synthetic_lookup -x "^(alloc::([a-z_]+::)+)Vec<.+>$" --category Rust
type summary add -F lldb_lookup.summary_lookup -x "^(alloc::([a-z_]+::)+)Vec<.+>$" --category Rust
```

With `__lldb_init_module`:

```python
def __lldb_init_module(debugger: SBDebugger, _dict: LLDBOpaque):
    # Ensure the category exists and is enabled
    rust_cat = debugger.GetCategory("Rust")

    if not rust_cat.IsValid():
        rust_cat = debugger.CreateCategory("Rust")

    rust_cat.SetEnabled(True)

    # Register Vec providers
    vec_regex = r"^(alloc::([a-z_]+::)+)Vec<.+>$"
    sb_name = lldb.SBTypeNameSpecifier(vec_regex, is_regex=True)

    sb_synth = lldb.SBTypeSynthetic.CreateWithClassName("lldb_lookup.VecSyntheticProvider")
    sb_synth.SetOptions(lldb.eTypeOptionCascade)

    sb_summary = lldb.SBTypeSummary.CreateWithFunctionName("lldb_lookup.VecSummaryProvider")
    sb_summary.SetOptions(lldb.eTypeOptionCascade)

    rust_cat.AddTypeSynthetic(sb_name, sb_synth)
    rust_cat.AddSummary(sb_name, sb_summary)
```

## Output

Without providers:

```text
(lldb) v vec_v
(alloc::vec::Vec<int, alloc::alloc::Global>) vec_v = {
  buf = {
    inner = {
      ptr = {
        pointer = (pointer = "\n")
        _marker = {}
      }
      cap = (__0 = 5)
      alloc = {}
    }
    _marker = {}
  }
  len = 5
}
(lldb) v vec_v[0]
error: <user expression 0>:1:6: subscripted value is not an array or pointer
   1 | vec_v[0]
     | ^
```

With providers (`v <var_name>` prints the summary and then a list of all children):

```text
(lldb) v vec_v
(Vec<int>) vec_v = vec![10, 20, 30, 40, 50] {
  [0] = 10
  [1] = 20
  [2] = 30
  [3] = 40
  [4] = 50
}
(lldb) v vec_v[0]
(int) vec_v[0] = 10
```

We can also confirm that the "hidden" length and capacity are still accessible:

```text
(lldb) v vec_v.len
(unsigned long long) vec_v.len = 5
(lldb) v vec_v.capacity
(unsigned long long) vec_v.capacity = 5
(lldb) v vec_v.cap
(unsigned long long) vec_v.cap = 5
```