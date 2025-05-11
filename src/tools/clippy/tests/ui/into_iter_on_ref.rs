#![allow(clippy::useless_vec, clippy::needless_borrow)]
#![warn(clippy::into_iter_on_ref)]

struct X;
use std::collections::*;

fn main() {
    for _ in &[1, 2, 3] {}
    for _ in vec![X, X] {}
    for _ in &vec![X, X] {}

    let _ = vec![1, 2, 3].into_iter();
    let _ = (&vec![1, 2, 3]).into_iter();
    //~^ into_iter_on_ref
    let _ = std::rc::Rc::from(&[X][..]).into_iter();
    //~^ into_iter_on_ref
    let _ = std::sync::Arc::from(&[X][..]).into_iter();
    //~^ into_iter_on_ref

    let _ = (&&&&&&&[1, 2, 3]).into_iter();
    //~^ into_iter_on_ref
    let _ = (&&&&mut &&&[1, 2, 3]).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut &mut &mut [1, 2, 3]).into_iter();
    //~^ into_iter_on_ref

    let _ = (&Some(4)).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut Some(5)).into_iter();
    //~^ into_iter_on_ref
    let _ = (&Ok::<_, i32>(6)).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut Err::<i32, _>(7)).into_iter();
    //~^ into_iter_on_ref
    let _ = (&Vec::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut Vec::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&BTreeMap::<i32, u64>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut BTreeMap::<i32, u64>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&VecDeque::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut VecDeque::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&LinkedList::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut LinkedList::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&HashMap::<i32, u64>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&mut HashMap::<i32, u64>::new()).into_iter();
    //~^ into_iter_on_ref

    let _ = (&BTreeSet::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&BinaryHeap::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = (&HashSet::<i32>::new()).into_iter();
    //~^ into_iter_on_ref
    let _ = std::path::Path::new("12/34").into_iter();
    //~^ into_iter_on_ref
    let _ = std::path::PathBuf::from("12/34").into_iter();
    //~^ into_iter_on_ref

    let _ = (&[1, 2, 3]).into_iter().next();
    //~^ into_iter_on_ref
}
