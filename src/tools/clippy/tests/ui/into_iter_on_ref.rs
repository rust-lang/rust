//@run-rustfix
#![allow(clippy::useless_vec, clippy::needless_borrow)]
#![warn(clippy::into_iter_on_ref)]

struct X;
use std::collections::*;

fn main() {
    for _ in &[1, 2, 3] {}
    for _ in vec![X, X] {}
    for _ in &vec![X, X] {}

    let _ = vec![1, 2, 3].into_iter();
    let _ = (&vec![1, 2, 3]).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = vec![1, 2, 3].into_boxed_slice().into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = std::rc::Rc::from(&[X][..]).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = std::sync::Arc::from(&[X][..]).into_iter(); //~ ERROR: equivalent to `.iter()

    let _ = (&&&&&&&[1, 2, 3]).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&&&&mut &&&[1, 2, 3]).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut &mut &mut [1, 2, 3]).into_iter(); //~ ERROR: equivalent to `.iter_mut()

    let _ = (&Some(4)).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut Some(5)).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&Ok::<_, i32>(6)).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut Err::<i32, _>(7)).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&Vec::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut Vec::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&BTreeMap::<i32, u64>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut BTreeMap::<i32, u64>::new()).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&VecDeque::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut VecDeque::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&LinkedList::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut LinkedList::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter_mut()
    let _ = (&HashMap::<i32, u64>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&mut HashMap::<i32, u64>::new()).into_iter(); //~ ERROR: equivalent to `.iter_mut()

    let _ = (&BTreeSet::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&BinaryHeap::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = (&HashSet::<i32>::new()).into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = std::path::Path::new("12/34").into_iter(); //~ ERROR: equivalent to `.iter()
    let _ = std::path::PathBuf::from("12/34").into_iter(); //~ ERROR: equivalent to `.iter()

    let _ = (&[1, 2, 3]).into_iter().next(); //~ ERROR: equivalent to `.iter()
}
