// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
// ignore-emscripten no threads support

#![feature(box_syntax)]

use std::thread;
use std::sync::mpsc::{channel, Sender};

#[derive(PartialEq, Debug)]
enum Message {
    Dropped,
    DestructorRan
}

struct SendOnDrop {
    sender: Sender<Message>
}

impl Drop for SendOnDrop {
    fn drop(&mut self) {
        self.sender.send(Message::Dropped).unwrap();
    }
}

enum Foo {
    SimpleVariant(Sender<Message>),
    NestedVariant(Box<usize>, SendOnDrop, Sender<Message>),
    FailingVariant { on_drop: SendOnDrop }
}

impl Drop for Foo {
    fn drop(&mut self) {
        match self {
            &mut Foo::SimpleVariant(ref mut sender) => {
                sender.send(Message::DestructorRan).unwrap();
            }
            &mut Foo::NestedVariant(_, _, ref mut sender) => {
                sender.send(Message::DestructorRan).unwrap();
            }
            &mut Foo::FailingVariant { .. } => {
                panic!("Failed");
            }
        }
    }
}

pub fn main() {
    let (sender, receiver) = channel();
    {
        let v = Foo::SimpleVariant(sender);
    }
    assert_eq!(receiver.recv().unwrap(), Message::DestructorRan);
    assert_eq!(receiver.recv().ok(), None);

    let (sender, receiver) = channel();
    {
        let v = Foo::NestedVariant(box 42, SendOnDrop { sender: sender.clone() }, sender);
    }
    assert_eq!(receiver.recv().unwrap(), Message::DestructorRan);
    assert_eq!(receiver.recv().unwrap(), Message::Dropped);
    assert_eq!(receiver.recv().ok(), None);

    let (sender, receiver) = channel();
    let t = thread::spawn(move|| {
        let v = Foo::FailingVariant { on_drop: SendOnDrop { sender: sender } };
    });
    assert_eq!(receiver.recv().unwrap(), Message::Dropped);
    assert_eq!(receiver.recv().ok(), None);
    drop(t.join());

    let (sender, receiver) = channel();
    let t = {
        thread::spawn(move|| {
            let mut v = Foo::NestedVariant(box 42, SendOnDrop {
                sender: sender.clone()
            }, sender.clone());
            v = Foo::NestedVariant(box 42,
                                   SendOnDrop { sender: sender.clone() },
                                   sender.clone());
            v = Foo::SimpleVariant(sender.clone());
            v = Foo::FailingVariant { on_drop: SendOnDrop { sender: sender } };
        })
    };
    assert_eq!(receiver.recv().unwrap(), Message::DestructorRan);
    assert_eq!(receiver.recv().unwrap(), Message::Dropped);
    assert_eq!(receiver.recv().unwrap(), Message::DestructorRan);
    assert_eq!(receiver.recv().unwrap(), Message::Dropped);
    assert_eq!(receiver.recv().unwrap(), Message::DestructorRan);
    assert_eq!(receiver.recv().unwrap(), Message::Dropped);
    assert_eq!(receiver.recv().ok(), None);
    drop(t.join());
}
