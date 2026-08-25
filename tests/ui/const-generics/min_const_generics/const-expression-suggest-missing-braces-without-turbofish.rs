fn foo<const C: usize>() {}

const BAR: usize = 42;

fn a() {
    foo<BAR + 3>(); //~ ERROR comparison operators cannot be chained
}
fn b() {
    foo<BAR + BAR>(); //~ ERROR comparison operators cannot be chained
}
fn c() {
    foo<3 + 3>(); //~ ERROR comparison operators cannot be chained
}
fn d() {
    foo<BAR - 3>(); //~ ERROR comparison operators cannot be chained
}
fn e() {
    foo<BAR - BAR>(); //~ ERROR comparison operators cannot be chained
}
fn f() {
    foo<100 - BAR>(); //~ ERROR comparison operators cannot be chained
}
fn g() {
    foo<bar<i32>()>(); //~ ERROR comparison operators cannot be chained
    //~^ ERROR expected one of `;` or `}`, found `>`
}
fn h() {
    foo<bar::<i32>()>(); //~ ERROR comparison operators cannot be chained
}
fn i() {
    foo<bar::<i32>() + BAR>(); //~ ERROR comparison operators cannot be chained
}
fn j() {
    foo<bar::<i32>() - BAR>(); //~ ERROR comparison operators cannot be chained
}
fn k() {
    foo<BAR - bar::<i32>()>(); //~ ERROR comparison operators cannot be chained
}
fn l() {
    foo<BAR - bar::<i32>()>(); //~ ERROR comparison operators cannot be chained
}

const fn bar<const C: usize>() -> usize {
    C
}

fn main() {}
