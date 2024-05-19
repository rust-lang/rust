use std::collections::hash_set::Iter;
use std::collections::HashSet;

fn iter_to_vec<'b, X>(i: Iter<'b, X>) -> Vec<X> {
    let i = i.map(|x| x.clone());
    i.collect() //~ ERROR E0277
}

fn main() {
    let scores = vec![(0, 0)]
        .iter()
        .map(|(a, b)| {
            a + b;
        });
    println!("{}", scores.sum::<i32>()); //~ ERROR E0277
    println!(
        "{}",
        vec![0, 1]
            .iter()
            .map(|x| x * 2)
            .map(|x| x as f64)
            .map(|x| x as i64)
            .filter(|x| *x > 0)
            .map(|x| { x + 1 })
            .map(|x| { x; })
            .sum::<i32>(), //~ ERROR E0277
    );
    println!(
        "{}",
        vec![0, 1]
            .iter()
            .map(|x| x * 2)
            .map(|x| x as f64)
            .filter(|x| *x > 0.0)
            .map(|x| { x + 1.0 })
            .sum::<i32>(), //~ ERROR E0277
    );
    println!("{}", vec![0, 1].iter().map(|x| { x; }).sum::<i32>()); //~ ERROR E0277
    println!("{}", vec![(), ()].iter().sum::<i32>()); //~ ERROR E0277
    let a = vec![0];
    let b = a.into_iter();
    let c = b.map(|x| x + 1);
    let d = c.filter(|x| *x > 10 );
    let e = d.map(|x| {
        x + 1;
    });
    let f = e.filter(|_| false);
    let g: Vec<i32> = f.collect(); //~ ERROR E0277

    let mut s = HashSet::new();
    s.insert(1u8);
    println!("{:?}", iter_to_vec(s.iter()));
}
