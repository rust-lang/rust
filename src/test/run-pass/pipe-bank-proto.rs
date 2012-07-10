// xfail-pretty

// An example of the bank protocol from eholk's blog post.
//
// http://theincredibleholk.wordpress.com/2012/07/06/rusty-pipes/

import pipes::try_recv;

type username = str;
type password = str;
type money = float;
type amount = float;

proto! bank {
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
}

fn macros() {
    #macro[
        [#move[x],
         unsafe { let y <- *ptr::addr_of(x); y }]
    ];
}

fn bank_client(+bank: bank::client::login) {
    import bank::*;

    let bank = client::login(bank, "theincredibleholk", "1234");
    let bank = alt try_recv(bank) {
      some(ok(connected)) {
        #move(connected)
      }
      some(invalid(_)) { fail "login unsuccessful" }
      none { fail "bank closed the connection" }
    };

    let bank = client::deposit(bank, 100.00);
    let bank = client::withdrawal(bank, 50.00);
    alt try_recv(bank) {
      some(money(m, _)) {
        io::println("Yay! I got money!");
      }
      some(insufficient_funds(_)) {
        fail "someone stole my money"
      }
      none {
        fail "bank closed the connection"
      }
    }
}

fn main() {
}