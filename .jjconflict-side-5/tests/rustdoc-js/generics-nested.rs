pub struct Out<A, B = ()> {
    a: A,
    b: B,
}

pub struct First<In = ()> {
    in_: In,
}

pub struct Second;

// Out<First<Second>>
pub fn alef() -> Out<First<Second>> {
    loop {}
}

pub fn bet() -> Out<First, Second> {
    loop {}
}
