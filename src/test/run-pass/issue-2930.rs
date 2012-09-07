proto! stream (
    stream:send<T:Send> {
        send(T) -> stream<T>
    }
)

fn main() {
    let (bc, _bp) = stream::init();

    stream::client::send(bc, ~"abc");
}
