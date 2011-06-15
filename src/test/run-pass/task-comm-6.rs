

fn main() { test00(); }

fn test00() {
    let int r = 0;
    let int sum = 0;
    let port[int] p = port();
    let chan[int] c0 = chan(p);
    let chan[int] c1 = chan(p);
    let chan[int] c2 = chan(p);
    let chan[int] c3 = chan(p);
    let int number_of_messages = 1000;
    let int i = 0;
    while (i < number_of_messages) {
        c0 <| i;
        c1 <| i;
        c2 <| i;
        c3 <| i;
        i += 1;
    }
    i = 0;
    while (i < number_of_messages) {
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