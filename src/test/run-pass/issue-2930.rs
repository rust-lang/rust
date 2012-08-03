// Generated from pipe compiler
mod stream {
    fn init<T: send>() -> (client::stream<T>, server::stream<T>) {
        pipes::entangle()
    }
    enum stream<T: send> { send(T, server::stream<T>), }
    mod client {
        fn send<T: send>(+pipe: stream<T>, +x_0: T) -> stream<T> {
            {
                let (c, s) = pipes::entangle();
                let message = stream::send(x_0, s);
                pipes::send(pipe, message);
                c
            }
        }
        type stream<T: send> = pipes::send_packet<stream::stream<T>>;
    }
    mod server {
        type stream<T: send> = pipes::recv_packet<stream::stream<T>>;
    }
}

fn main() {
    let (bc, _bp) = stream::init();

    stream::client::send(bc, ~"abc");
}
