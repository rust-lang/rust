trait TypeMapper {
    type MapType;
}

type Mapped<T> = <T as TypeMapper>::MapType;

struct Test {}

impl TypeMapper for () {
    type MapType = Test;
}

fn test() {
    Mapped::<()> {};
    None; //~ ERROR type annotations needed
}

fn main() {}
