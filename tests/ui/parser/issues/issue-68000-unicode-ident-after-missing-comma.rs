pub struct Foo {
    pub bar: Vec<i32>ö
    //~^ ERROR expected `,`, or `}`, found `ö`
} //~ ERROR expected `:`, found `}`

fn main() {}
