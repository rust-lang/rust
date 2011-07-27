

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p: port[int] = port();
    let c: chan[int] = chan(p);
    let number_of_messages: int = 1000;
    let i: int = 0;
    while i < number_of_messages { c <| i; i += 1; }
    i = 0;
    while i < number_of_messages { p |> r; sum += r; i += 1; }
    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}