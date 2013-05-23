// xfail-fast
// xfail-test

// XFAIL'd because this is going to be revamped, and it's not compatible as
// written with the new mutability rules.

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Examples from Eric's internship final presentation.
//
// Code is easier to write in emacs, and it's good to be sure all the
// code samples compile (or not) as they should.

use double_buffer::client::*;
use double_buffer::give_buffer;
use std::comm::Selectable;

macro_rules! select_if (
    {
        $index:expr,
        $count:expr,
        $port:path => [
            $($message:path$(($($x: ident),+))dont_type_this*
              -> $next:ident $e:expr),+
        ],
        $( $ports:path => [
            $($messages:path$(($($xs: ident),+))dont_type_this*
              -> $nexts:ident $es:expr),+
        ], )*
    } => {
        if $index == $count {
            match std::pipes::try_recv($port) {
              $(Some($message($($($x,)+)* next)) => {
                let $next = next;
                $e
              })+
              _ => fail!()
            }
        } else {
            select_if!(
                $index,
                $count + 1,
                $( $ports => [
                    $($messages$(($($xs),+))dont_type_this*
                      -> $nexts $es),+
                ], )*
            )
        }
    };

    {
        $index:expr,
        $count:expr,
    } => {
        fail!()
    }
)

macro_rules! select (
    {
        $( $port:path => {
            $($message:path$(($($x: ident),+))dont_type_this*
              -> $next:ident $e:expr),+
        } )+
    } => ({
        let index = std::comm::selecti([$(($port).header()),+]);
        select_if!(index, 0, $( $port => [
            $($message$(($($x),+))dont_type_this* -> $next $e),+
        ], )+)
    })
)

// Types and protocols
pub struct Buffer {
    foo: (),

}

impl Drop for Buffer {
    fn finalize(&self) {}
}

proto! double_buffer (
    acquire:send {
        request -> wait_buffer
    }

    wait_buffer:recv {
        give_buffer(::Buffer) -> release
    }

    release:send {
        release(::Buffer) -> acquire
    }
)

// Code examples
fn render(_buffer: &Buffer) {
    // A dummy function.
}

fn draw_frame(+channel: double_buffer::client::acquire) {
    let channel = request(channel);
    select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
                release(channel, buffer)
            }
        }
    );
}

fn draw_two_frames(+channel: double_buffer::client::acquire) {
    let channel = request(channel);
    let channel = select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
                release(channel, buffer)
            }
        }
    );
    let channel = request(channel);
    select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
                release(channel, buffer)
            }
        }
    );
}

#[cfg(bad1)]
fn draw_two_frames_bad1(+channel: double_buffer::client::acquire) {
    let channel = request(channel);
    select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
            }
        }
    );
    let channel = request(channel);
    select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
                release(channel, buffer)
            }
        }
    );
}

#[cfg(bad2)]
fn draw_two_frames_bad2(+channel: double_buffer::client::acquire) {
    let channel = request(channel);
    select! (
        channel => {
            give_buffer(buffer) -> channel {
                render(&buffer);
                release(channel, buffer);
                render(&buffer);
                release(channel, buffer);
            }
        }
    );
}

pub fn main() { }
