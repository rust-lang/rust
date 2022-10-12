// run-pass

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
    fn option_loud_drop(&self, n: u32) -> Option<LoudDrop> {
        Some(LoudDrop(self, n))
    }

    fn loud_drop(&self, n: u32) -> LoudDrop {
        LoudDrop(self, n)
    }

    fn print(&self, n: u32) {
        println!("{}", n);
        self.0.borrow_mut().push(n)
    }

    fn if_(&self) {
        if self.option_loud_drop(1).is_some() {
            self.print(2);
        }

        if self.option_loud_drop(3).is_none() {
            unreachable!();
        } else if self.option_loud_drop(4).is_some() {
            self.print(5);
        }

        if {
            if self.option_loud_drop(7).is_some() && self.option_loud_drop(6).is_some() {
                self.loud_drop(8);
                true
            } else {
                false
            }
        } {
            self.print(9);
        }
    }

    fn if_let(&self) {
        if let None = self.option_loud_drop(2) {
            unreachable!();
        } else {
            self.print(1);
        }

        if let Some(_) = self.option_loud_drop(4) {
            self.print(3);
        }

        if let Some(_d) = self.option_loud_drop(6) {
            self.print(5);
        }
    }

    fn match_(&self) {
        match self.option_loud_drop(2) {
            _any => self.print(1),
        }

        match self.option_loud_drop(4) {
            _ => self.print(3),
        }

        match self.option_loud_drop(6) {
            Some(_) => self.print(5),
            _ => unreachable!(),
        }

        match {
            let _ = self.loud_drop(7);
            let _d = self.loud_drop(9);
            self.print(8);
            ()
        } {
            () => self.print(10),
        }

        match {
            match self.option_loud_drop(14) {
                _ => {
                    self.print(11);
                    self.option_loud_drop(13)
                }
            }
        } {
            _ => self.print(12),
        }

        match {
            loop {
                break match self.option_loud_drop(16) {
                    _ => {
                        self.print(15);
                        self.option_loud_drop(18)
                    }
                };
            }
        } {
            _ => self.print(17),
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
    println!("-- if --");
    let collector = DropOrderCollector::default();
    collector.if_();
    collector.assert_sorted();

    println!("-- if let --");
    let collector = DropOrderCollector::default();
    collector.if_let();
    collector.assert_sorted();

    println!("-- match --");
    let collector = DropOrderCollector::default();
    collector.match_();
    collector.assert_sorted();
}
