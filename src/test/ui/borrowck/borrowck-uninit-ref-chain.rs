// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

struct S<X, Y> {
    x: X,
    y: Y,
}

fn main() {
    let x: &&Box<i32>;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]

    let x: &&S<i32, i32>;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]

    let x: &&i32;
    let _y = &**x; //[ast]~ ERROR use of possibly uninitialized variable: `**x` [E0381]
                   //[mir]~^ [E0381]


    let mut a: S<i32, i32>;
    a.x = 0;       //[mir]~ ERROR assign to part of possibly uninitialized variable: `a` [E0381]
    let _b = &a.x; //[ast]~ ERROR use of possibly uninitialized variable: `a.x` [E0381]


    let mut a: S<&&i32, &&i32>;
    a.x = &&0;       //[mir]~ ERROR assign to part of possibly uninitialized variable: `a` [E0381]
    let _b = &**a.x; //[ast]~ ERROR use of possibly uninitialized variable: `**a.x` [E0381]



    let mut a: S<i32, i32>;
    a.x = 0;       //[mir]~ ERROR assign to part of possibly uninitialized variable: `a` [E0381]
    let _b = &a.y; //[ast]~ ERROR use of possibly uninitialized variable: `a.y` [E0381]


    let mut a: S<&&i32, &&i32>;
    a.x = &&0;       //[mir]~ assign to part of possibly uninitialized variable: `a` [E0381]
    let _b = &**a.y; //[ast]~ ERROR use of possibly uninitialized variable: `**a.y` [E0381]

}
