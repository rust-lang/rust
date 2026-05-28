type A = for<'a + 'b> fn(); //~ ERROR expected one of `,`, `:`, or `>`, found `+`

fn main() {}
