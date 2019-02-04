mod a {
    pub struct A(usize, usize);
    pub struct B {a: usize, b: usize}

}

fn main(){
    let x = a::A(3, 4);
    //~^ ERROR fields of struct `a::A` are private
    let x = a::A;
    //~^ ERROR tuple struct `a::A` is private
    let x = a::B {a:1, b:2};
    //~^ ERROR fields of struct `a::B` are private
}

fn foo(_x: a::A) {} // ok
fn bar(_x: a::B) {} // ok
