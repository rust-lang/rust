//@ run-pass
//@ compile-flags: -Z validate-mir
//@ edition: 2024

use std::cell::RefCell;
use std::convert::TryInto;

#[derive(Default)]
struct DropOrderCollector(RefCell<Vec<u32>>);

struct LoudDrop<'a>(&'a DropOrderCollector, u32);

impl Drop for LoudDrop<'_> {
    fn drop(&mut self) {
        println!("{}", self.1);
        self.0.0.borrow_mut().push(self.1);
    }
}

impl DropOrderCollector {
    fn option_loud_drop(&self, n: u32) -> Option<LoudDrop<'_>> {
        Some(LoudDrop(self, n))
    }

    fn print(&self, n: u32) {
        println!("{}", n);
        self.0.borrow_mut().push(n)
    }

    fn let_chain(&self) {
        // take the "then" branch
        if self.option_loud_drop(1).is_some() // 1
            && self.option_loud_drop(2).is_some() // 2
            && let Some(_d) = self.option_loud_drop(4) { // 4
            self.print(3); // 3
        }

        // take the "else" branch
        if self.option_loud_drop(5).is_some() // 1
            && self.option_loud_drop(6).is_some() // 2
            && let None = self.option_loud_drop(7) { // 4
            unreachable!();
        } else {
            self.print(8); // 3
        }

        // let exprs interspersed
        if self.option_loud_drop(9).is_some() // 1
            && let Some(_d) = self.option_loud_drop(13) // 5
            && self.option_loud_drop(10).is_some() // 2
            && let Some(_e) = self.option_loud_drop(12) { // 4
            self.print(11); // 3
        }

        // let exprs first
        if let Some(_d) = self.option_loud_drop(18) // 5
            && let Some(_e) = self.option_loud_drop(17) // 4
            && self.option_loud_drop(14).is_some() // 1
            && self.option_loud_drop(15).is_some() { // 2
                self.print(16); // 3
            }

        // let exprs last
        if self.option_loud_drop(19).is_some() // 1
            && self.option_loud_drop(20).is_some() // 2
            && let Some(_d) = self.option_loud_drop(23) // 5
            && let Some(_e) = self.option_loud_drop(22) { // 4
                self.print(21); // 3
        }
    }

    fn while_(&self) {
        let mut v = self.option_loud_drop(4);
        while let Some(_d) = v
            && self.option_loud_drop(1).is_some()
            && self.option_loud_drop(2).is_some() {
            self.print(3);
            v = None;
        }
    }

    fn assert_sorted(self) {
        assert!(
            self.0
                .into_inner()
                .into_iter()
                .enumerate()
                .all(|(idx, item)| idx + 1 == item.try_into().unwrap())
        );
    }
}

fn main() {
    println!("-- let chain --");
    let collector = DropOrderCollector::default();
    collector.let_chain();
    collector.assert_sorted();

    println!("-- while --");
    let collector = DropOrderCollector::default();
    collector.while_();
    collector.assert_sorted();
}
