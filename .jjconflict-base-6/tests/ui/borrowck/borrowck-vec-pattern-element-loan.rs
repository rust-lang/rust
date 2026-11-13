fn a<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec;
    let tail = match vec {
        &[_, ref tail @ ..] => tail,
        _ => panic!("a")
    };
    tail //~ ERROR cannot return value referencing local variable `vec`
}

fn b<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec;
    let init = match vec {
        &[ref init @ .., _] => init,
        _ => panic!("b")
    };
    init //~ ERROR cannot return value referencing local variable `vec`
}

fn c<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec;
    let slice = match vec {
        &[_, ref slice @ .., _] => slice,
        _ => panic!("c")
    };
    slice //~ ERROR cannot return value referencing local variable `vec`
}

fn main() {}
