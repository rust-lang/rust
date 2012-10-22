proto! stream (
    Stream:send<T:Send> {
        send(T) -> Stream<T>
    }
)

fn main() {
    let (bc, _bp) = stream::init();

    stream::client::send(move bc, ~"abc");
}
