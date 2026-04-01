// Do not try to evaluate static initalizers that reference
// ill-defined types. This used to be an ICE.
// See issues #125842 and #124464.
struct Struct {
    field: Option<u8>,
    field: u8,
//~^ ERROR field `field` is already declared
}

static STATIC_A: Struct = Struct {
    field: 1
};

static STATIC_B: Struct = {
    let field = 1;
    Struct {
        field,
    }
};

fn main() {}
