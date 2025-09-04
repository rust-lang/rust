# TypeTrees for Autodiff

## What are TypeTrees?
Memory layout descriptors for Enzyme. Tell Enzyme exactly how types are structured in memory so it can compute derivatives efficiently.

## Structure
```rust
TypeTree(Vec<Type>)

Type {
    offset: isize,  // byte offset (-1 = everywhere)
    size: usize,    // size in bytes
    kind: Kind,     // Float, Integer, Pointer, etc.
    child: TypeTree // nested structure
}
```

## Example: `fn compute(x: &f32, data: &[f32]) -> f32`

**Input 0: `x: &f32`**
```rust
TypeTree(vec![Type {
    offset: -1, size: 8, kind: Pointer,
    child: TypeTree(vec![Type {
        offset: -1, size: 4, kind: Float,
        child: TypeTree::new()
    }])
}])
```

**Input 1: `data: &[f32]`**
```rust
TypeTree(vec![Type {
    offset: -1, size: 8, kind: Pointer,
    child: TypeTree(vec![Type {
        offset: -1, size: 4, kind: Float,  // -1 = all elements
        child: TypeTree::new()
    }])
}])
```

**Output: `f32`**
```rust
TypeTree(vec![Type {
    offset: -1, size: 4, kind: Float,
    child: TypeTree::new()
}])
```

## Why Needed?
- Enzyme can't deduce complex type layouts from LLVM IR
- Prevents slow memory pattern analysis
- Enables correct derivative computation for nested structures
- Tells Enzyme which bytes are differentiable vs metadata

## What Enzyme Does With This Information:

Without TypeTrees:
```llvm
; Enzyme sees generic LLVM IR:
define float @distance(i8* %p1, i8* %p2) {
; Has to guess what these pointers point to
; Slow analysis of all memory operations
; May miss optimization opportunities
}
```

With TypeTrees:
```llvm
define "enzyme_type"="{[]:Float@float}" float @distance(
    ptr "enzyme_type"="{[]:Pointer}" %p1, 
    ptr "enzyme_type"="{[]:Pointer}" %p2
) {
; Enzyme knows exact type layout
; Can generate efficient derivative code directly
}
```

# TypeTrees - Offset and -1 Explained

## Type Structure

```rust
Type {
    offset: isize, // WHERE this type starts
    size: usize,   // HOW BIG this type is
    kind: Kind,    // WHAT KIND of data (Float, Int, Pointer)
    child: TypeTree // WHAT'S INSIDE (for pointers/containers)
}
```

## Offset Values

### Regular Offset (0, 4, 8, etc.)
**Specific byte position within a structure**

```rust
struct Point {
    x: f32, // offset 0, size 4
    y: f32, // offset 4, size 4
    id: i32, // offset 8, size 4
}
```

TypeTree for `&Point` (internal representation):
```rust
TypeTree(vec![
    Type { offset: 0, size: 4, kind: Float },   // x at byte 0
    Type { offset: 4, size: 4, kind: Float },   // y at byte 4
    Type { offset: 8, size: 4, kind: Integer }  // id at byte 8
])
```

Generates LLVM:
```llvm
"enzyme_type"="{[]:Float@float}"
```

### Offset -1 (Special: "Everywhere")
**Means "this pattern repeats for ALL elements"**

#### Example 1: Array `[f32; 100]`
```rust
TypeTree(vec![Type {
    offset: -1, // ALL positions
    size: 4,    // each f32 is 4 bytes
    kind: Float, // every element is float
}])
```

Instead of listing 100 separate Types with offsets `0,4,8,12...396`

#### Example 2: Slice `&[i32]`
```rust
// Pointer to slice data
TypeTree(vec![Type {
    offset: -1, size: 8, kind: Pointer,
    child: TypeTree(vec![Type {
        offset: -1, // ALL slice elements
        size: 4,    // each i32 is 4 bytes
        kind: Integer
    }])
}])
```

#### Example 3: Mixed Structure
```rust
struct Container {
    header: i64,        // offset 0
    data: [f32; 1000],  // offset 8, but elements use -1
}
```

```rust
TypeTree(vec![
    Type { offset: 0, size: 8, kind: Integer }, // header
    Type { offset: 8, size: 4000, kind: Pointer,
        child: TypeTree(vec![Type {
            offset: -1, size: 4, kind: Float // ALL array elements
        }])
    }
])
```