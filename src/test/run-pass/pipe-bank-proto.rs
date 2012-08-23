// xfail-pretty

// An example of the bank protocol from eholk's blog post.
//
// http://theincredibleholk.wordpress.com/2012/07/06/rusty-pipes/

import pipes::try_recv;

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
    { $x:expr } => { unsafe { let y <- *ptr::addr_of($x); y } }
)

fn switch<T: send, U>(+endp: pipes::recv_packet<T>,
                      f: fn(+option<T>) -> U) -> U {
    f(pipes::try_recv(endp))
}

fn move_it<T>(-x: T) -> T { x }

macro_rules! follow (
    {
        $($message:path$(($($x: ident),+))||* -> $next:ident $e:expr)+
    } => (
        |m| match move m {
          $(some($message($($($x,)+)* next)) => {
            let $next = move_it!(next);
            $e })+
          _ => { fail }
        }
    );
)

fn client_follow(+bank: bank::client::login) {
    import bank::*;

    let bank = client::login(bank, ~"theincredibleholk", ~"1234");
    let bank = switch(bank, follow! (
        ok -> connected { connected }
        invalid -> _next { fail ~"bank closed the connected" }
    ));

    let bank = client::deposit(bank, 100.00);
    let bank = client::withdrawal(bank, 50.00);
    switch(bank, follow! (
        money(m) -> _next {
            io::println(~"Yay! I got money!");
        }
        insufficient_funds -> _next {
            fail ~"someone stole my money"
        }
    ));
}

fn bank_client(+bank: bank::client::login) {
    import bank::*;

    let bank = client::login(bank, ~"theincredibleholk", ~"1234");
    let bank = match try_recv(bank) {
      some(ok(connected)) => {
        move_it!(connected)
      }
      some(invalid(_)) => { fail ~"login unsuccessful" }
      none => { fail ~"bank closed the connection" }
    };

    let bank = client::deposit(bank, 100.00);
    let bank = client::withdrawal(bank, 50.00);
    match try_recv(bank) {
      some(money(m, _)) => {
        io::println(~"Yay! I got money!");
      }
      some(insufficient_funds(_)) => {
        fail ~"someone stole my money"
      }
      none => {
        fail ~"bank closed the connection"
      }
    }
}

fn main() {
}
