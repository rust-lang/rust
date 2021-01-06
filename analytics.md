# file name change

change module system and `use` clauses.

This means we would not change all possiable places for now.(I think this is hard to implement even in future.)

## module system

### physical module and logic module

each file and folder(lib.rs in the folder) is a physical module.

`use` and `pub` could import and export module as logic.

logic module could be seen as `namespace`.

Only logic module matters.

### module tree

only update the module which is included in the main branch.

### module declaration

> logic module could be defined in its ancestor's folder.

> module

## use clauses

## NOTE

1. path attribute
if some module is declared by `#[path = "filePath"]`, only update the `filePath`.

2. not update `mod.rs` change, like to `mod1.rs`

3. for folder name change(not add more subfolders), equal to change its `mod.rs`.

4. However, mod could be declared through `mod XX {}` in its ancestor file.

5. Key: How to update? Through semantic tree, or plain text? I worry the former one is not that clever.
