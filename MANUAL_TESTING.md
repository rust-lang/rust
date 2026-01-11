# Manual Testing Instructions for Field Reordering Suggestion

## Overview
This PR adds a diagnostic suggestion to reorder struct fields when a field moves a value and a subsequent field tries to borrow from it.

## Prerequisites
1. Build the Rust compiler with your changes:
   ```bash
   ./x.py build --stage 1
   ```
   Note: This may take 30+ minutes on first build.

## Test Case 1: Basic Field Reordering (Original Issue #150320)

### Create test file `test_basic.rs`:
```rust
struct Foo {
    a: String,
    b: usize,
}

impl Foo {
    fn new(a: String) -> Self {
        Self {
            a,
            b: a.len(),
        }
    }
}

fn main() {
    let _ = Foo::new("hello".to_string());
}
```

### Compile and verify the error message:
```bash
./build/x86_64-unknown-linux-gnu/stage1/bin/rustc test_basic.rs
```

### Expected Output:
```
error[E0382]: borrow of moved value: `a`
  --> test_basic.rs:10:16
   |
 8 |     fn new(a: String) -> Self {
   |            - move occurs because `a` has type `String`, which does not implement the `Copy` trait
 9 |         Self {
10 |             a,
   |             - value moved here
11 |             b: a.len(),
   |                ^ value borrowed here after move
   |
help: consider initializing `b` before `a`
   |
LL ~             b: a.len(),
LL ~             a,
   |
help: consider cloning the value if the performance cost is acceptable
   |
10 |             a: a.clone(),
   |              +++++++++++
```

### Verify:
- ✅ The first suggestion says "consider initializing `b` before `a`" (using actual field names)
- ✅ The suggestion shows swapping the field order
- ✅ The clone suggestion appears as a fallback

## Test Case 2: Different Variable Names

### Create test file `test_different_names.rs`:
```rust
struct Config {
    path: String,
    length: usize,
}

impl Config {
    fn new(path: String) -> Self {
        Self {
            path,
            length: path.len(),
        }
    }
}

fn main() {
    let _ = Config::new("/tmp/file".to_string());
}
```

### Compile:
```bash
./build/x86_64-unknown-linux-gnu/stage1/bin/rustc test_different_names.rs
```

### Expected Output:
```
help: consider initializing `length` before `path`
   |
LL ~             length: path.len(),
LL ~             path,
   |
```

### Verify:
- ✅ The suggestion uses "length" and "path" (not hardcoded "a" and "b")
- ✅ Field names match the actual struct definition

## Test Case 3: Multiple Fields

### Create test file `test_multiple.rs`:
```rust
struct Data {
    name: String,
    count: usize,
    size: usize,
}

impl Data {
    fn new(name: String) -> Self {
        Self {
            name,
            count: name.len(),
            size: name.capacity(),
        }
    }
}

fn main() {
    let _ = Data::new("test".to_string());
}
```

### Compile:
```bash
./build/x86_64-unknown-linux-gnu/stage1/bin/rustc test_multiple.rs
```

### Verify:
- ✅ Suggestions appear for both `count` and `size` fields
- ✅ Each suggestion uses the correct field names

## Test Case 4: No Suggestion When Order is Correct

### Create test file `test_correct_order.rs`:
```rust
struct Foo {
    b: usize,
    a: String,
}

impl Foo {
    fn new(a: String) -> Self {
        Self {
            b: a.len(),  // Borrow happens first
            a,           // Move happens after - OK!
        }
    }
}

fn main() {
    let _ = Foo::new("hello".to_string());
}
```

### Compile:
```bash
./build/x86_64-unknown-linux-gnu/stage1/bin/rustc test_correct_order.rs
```

### Expected:
- ✅ No errors - compiles successfully
- ✅ No reordering suggestion needed

## Running Automated Tests

### Run the specific test:
```bash
./x.py test tests/ui/borrowck/issue-150320-suggest-field-reordering.rs --stage 1
```

### Run all borrowck diagnostic tests:
```bash
./x.py test tests/ui/borrowck --stage 1
```

## Success Criteria

All of the following should be true:
1. ✅ Field reordering suggestion appears as the PRIMARY suggestion (before clone)
2. ✅ Suggestion uses ACTUAL field names from the code (not hardcoded "a" and "b")
3. ✅ Suggestion is machine-applicable (can be auto-applied)
4. ✅ No suggestion when fields are already in the correct order
5. ✅ All automated tests pass
6. ✅ No regression in other borrowck tests
