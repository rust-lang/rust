// Regression test for #154568
//@ compile-flags: -Znext-solver=globally

trait Role {
    type Inner;
}

struct HandshakeCallback<C>(C);
struct Handshake<R: Role>(R::Inner);

fn main() {
    let callback = HandshakeCallback(());
    let handshake = Handshake(callback.0.clone());
    //~^ ERROR type annotations needed
    match &handshake {
        hs if (|| {
            let borrowed_inner = &hs.0;
            borrowed_inner == &callback.0
        })() => println!(),
        _ => {}
    }
}
