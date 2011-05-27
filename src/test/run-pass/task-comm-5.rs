fn main() -> () {
   test00();
}

fn test00() {
    let int r = 0;    
    let int sum = 0;
    let port[int] p = port();
    let chan[int] c = chan(p);
    let int number_of_messages = 1000;

    let int i = 0;
    while (i < number_of_messages) {
        c <| i;
        i += 1;
    }

    i = 0;
    while (i < number_of_messages) {
        p |> r; sum += r;
        i += 1;
    }
    
    assert (sum == ((number_of_messages * (number_of_messages - 1)) / 2));
}