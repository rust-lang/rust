

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p: port[int] = port();
    let c0: chan[int] = chan(p);
    let c1: chan[int] = chan(p);
    let c2: chan[int] = chan(p);
    let c3: chan[int] = chan(p);
    let number_of_messages: int = 1000;
    let i: int = 0;
    while i < number_of_messages {
        c0 <| i;
        c1 <| i;
        c2 <| i;
        c3 <| i;
        i += 1;
    }
    i = 0;
    while i < number_of_messages {
        p |> r;
        sum += r;
        p |> r;
        sum += r;
        p |> r;
        sum += r;
        p |> r;
        sum += r;
        i += 1;
    }
    assert (sum == 1998000);
    // assert (sum == 4 * ((number_of_messages *
    //                   (number_of_messages - 1)) / 2));

}