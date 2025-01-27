// use core::iter::*;

// FIXME: need to implement PeekableIterator for Cloned
// #[test]
// fn test_peekable_iterator() {
//     let xs = vec![0, 1, 2, 3, 4, 5];

//     let mut it = xs.iter().cloned();
//     assert_eq!(it.len(), 6);
//     assert_eq!(it.peek().unwrap(), &0);
//     assert_eq!(it.len(), 6);
//     assert_eq!(it.next().unwrap(), 0);
//     assert_eq!(it.len(), 5);
//     assert_eq!(it.next().unwrap(), 1);
//     assert_eq!(it.len(), 4);
//     assert_eq!(it.next().unwrap(), 2);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.peek().unwrap(), &3);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.peek().unwrap(), &3);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.next().unwrap(), 3);
//     assert_eq!(it.len(), 2);
//     assert_eq!(it.next().unwrap(), 4);
//     assert_eq!(it.len(), 1);
//     assert_eq!(it.peek().unwrap(), &5);
//     assert_eq!(it.len(), 1);
//     assert_eq!(it.next().unwrap(), 5);
//     assert_eq!(it.len(), 0);
//     assert!(it.peek().is_none());
//     assert_eq!(it.len(), 0);
//     assert!(it.next().is_none());
//     assert_eq!(it.len(), 0);

//     let mut it = xs.iter().cloned();
//     assert_eq!(it.len(), 6);
//     assert_eq!(it.peek().unwrap(), &0);
//     assert_eq!(it.len(), 6);
//     assert_eq!(it.next_back().unwrap(), 5);
//     assert_eq!(it.len(), 5);
//     assert_eq!(it.next_back().unwrap(), 4);
//     assert_eq!(it.len(), 4);
//     assert_eq!(it.next_back().unwrap(), 3);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.peek().unwrap(), &0);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.peek().unwrap(), &0);
//     assert_eq!(it.len(), 3);
//     assert_eq!(it.next_back().unwrap(), 2);
//     assert_eq!(it.len(), 2);
//     assert_eq!(it.next_back().unwrap(), 1);
//     assert_eq!(it.len(), 1);
//     assert_eq!(it.peek().unwrap(), &0);
//     assert_eq!(it.len(), 1);
//     assert_eq!(it.next_back().unwrap(), 0);
//     assert_eq!(it.len(), 0);
//     assert!(it.peek().is_none());
//     assert_eq!(it.len(), 0);
//     assert!(it.next_back().is_none());
//     assert_eq!(it.len(), 0);
// }
