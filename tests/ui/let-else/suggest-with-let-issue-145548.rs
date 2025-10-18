// `Some(x) = Some(1)` is treated as a let statement/expr,
// so we should not suggest to add `let ... else{...}`

fn foo1(){
    let x = 2;
    Some(x) = Some(1); //~ ERROR refutable pattern in local binding [E0005]
}

fn foo2(){
    let x = 2;
    let Some(x) = Some(1); //~ ERROR refutable pattern in local binding [E0005]
}

fn foo3(){
    Some(()) = Some(()); //~ ERROR refutable pattern in local binding [E0005]
}

fn main() {}
