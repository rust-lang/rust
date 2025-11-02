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
        offset: 0, size: 4, kind: Float,  // Single value: use offset 0
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
    offset: 0, size: 4, kind: Float,  // Single scalar: use offset 0
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
define float @distance(ptr %p1, ptr %p2) {
; Has to guess what these pointers point to
; Slow analysis of all memory operations
; May miss optimization opportunities
}
```

With TypeTrees:
```llvm
define "enzyme_type"="{[-1]:Float@float}" float @distance(
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float}" %p1, 
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float}" %p2
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

Generates LLVM
```llvm
"enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer}"
```

### Offset -1 (Special: "Everywhere")
**Means "this pattern repeats for ALL elements"**

#### Example 1: Direct Array `[f32; 100]` (no pointer indirection)
```rust
TypeTree(vec![Type {
    offset: -1, // ALL positions
    size: 4,    // each f32 is 4 bytes
    kind: Float, // every element is float
}])
```

Generates LLVM: `"enzyme_type"="{[-1]:Float@float}"`

#### Example 1b: Array Reference `&[f32; 100]` (with pointer indirection)  
```rust
TypeTree(vec![Type {
    offset: -1, size: 8, kind: Pointer,
    child: TypeTree(vec![Type {
        offset: -1, // ALL array elements
        size: 4,    // each f32 is 4 bytes
        kind: Float, // every element is float
    }])
}])
```

Generates LLVM: `"enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}"`

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

Generates LLVM: `"enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}"`

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

## Key Distinction: Single Values vs Arrays

**Single Values** use offset `0` for precision:
- `&f32` has exactly one f32 value at offset 0
- More precise than using -1 ("everywhere")  
- Generates: `{[-1]:Pointer, [-1,0]:Float@float}`

**Arrays** use offset `-1` for efficiency:
- `&[f32; 100]` has the same pattern repeated 100 times
- Using -1 avoids listing 100 separate offsets
- Generates: `{[-1]:Pointer, [-1,-1]:Float@float}`