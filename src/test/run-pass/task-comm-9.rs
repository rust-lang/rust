impure fn main() -> () {
   test00();
}

impure fn test00_start(chan[int] c, int number_of_messages) {
    let int i = 0;
    while (i < number_of_messages) {
        c <| i;
        i += 1;
    }    
}

impure fn test00() {
    let int r = 0;    
    let int sum = 0;
    let port[int] p = port();
    let int number_of_messages = 10;
        
    let task t0 = spawn thread "child"
        test00_start(chan(p), number_of_messages);
    
    let int i = 0;
    while (i < number_of_messages) {
        r <- p; sum += r; log (r);
        i += 1;
    }
            
    join t0;
    
    check (sum == (number_of_messages * (number_of_messages - 1)) / 2);
}