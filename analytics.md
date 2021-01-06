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

### module declaration

> logic module could be defined in its ancestor's folder.

> module

## use clauses

