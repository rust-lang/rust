// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// An example of the bank protocol from eholk's blog post.
//
// http://theincredibleholk.wordpress.com/2012/07/06/rusty-pipes/

use pipes::try_recv;

type username = ~str;
type password = ~str;
type money = float;
type amount = float;

proto! bank (
    login:send {
        login(username, password) -> login_response
    }

    login_response:recv {
        ok -> connected,
        invalid -> login
    }

    connected:send {
        deposit(money) -> connected,
        withdrawal(amount) -> withdrawal_response
    }

    withdrawal_response:recv {
        money(money) -> connected,
        insufficient_funds -> connected
    }
)

macro_rules! move_it (
    { $x:expr } => { unsafe { let y = move *ptr::addr_of(&($x)); move y } }
)

fn switch<T: Send, U>(+endp: pipes::RecvPacket<T>,
                      f: fn(+v: Option<T>) -> U) -> U {
    f(pipes::try_recv(move endp))
}

fn move_it<T>(-x: T) -> T { move x }

macro_rules! follow (
    {
        $($message:path$(($($x: ident),+))||* -> $next:ident $e:expr)+
    } => (
        |m| match move m {
          $(Some($message($($($x,)+)* next)) => {
            let $next = move_it!(next);
            $e })+
          _ => { fail }
        }
    );
)

fn client_follow(+bank: bank::client::login) {
    use bank::*;

    let bank = client::login(move bank, ~"theincredibleholk", ~"1234");
    let bank = switch(move bank, follow! (
        ok -> connected { move connected }
        invalid -> _next { fail ~"bank closed the connected" }
    ));

    let bank = client::deposit(move bank, 100.00);
    let bank = client::withdrawal(move bank, 50.00);
    switch(move bank, follow! (
        money(m) -> _next {
            io::println(~"Yay! I got money!");
        }
        insufficient_funds -> _next {
            fail ~"someone stole my money"
        }
    ));
}

fn bank_client(+bank: bank::client::login) {
    use bank::*;

    let bank = client::login(move bank, ~"theincredibleholk", ~"1234");
    let bank = match try_recv(move bank) {
      Some(ok(connected)) => {
        move_it!(connected)
      }
      Some(invalid(_)) => { fail ~"login unsuccessful" }
      None => { fail ~"bank closed the connection" }
    };

    let bank = client::deposit(move bank, 100.00);
    let bank = client::withdrawal(move bank, 50.00);
    match try_recv(move bank) {
      Some(money(*)) => {
        io::println(~"Yay! I got money!");
      }
      Some(insufficient_funds(_)) => {
        fail ~"someone stole my money"
      }
      None => {
        fail ~"bank closed the connection"
      }
    }
}

fn main() {
}
