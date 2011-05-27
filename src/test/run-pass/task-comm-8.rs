// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() -> () {
   test00();
}

fn test00_start(chan[int] c, int start, int number_of_messages) {
    let int i = 0;
    while (i < number_of_messages) {
        c <| start + i;
        i += 1;
    }    
}

fn test00() {
    let int r = 0;    
    let int sum = 0;
    let port[int] p = port();
    let int number_of_messages = 10;
        
    let task t0 = spawn thread test00_start(chan(p), 
                               number_of_messages * 0, number_of_messages);
    let task t1 = spawn thread test00_start(chan(p), 
                               number_of_messages * 1, number_of_messages);
    let task t2 = spawn thread test00_start(chan(p), 
                               number_of_messages * 2, number_of_messages);
    let task t3 = spawn thread test00_start(chan(p), 
                               number_of_messages * 3, number_of_messages);
    
    let int i = 0;
    while (i < number_of_messages) {
        p |> r; sum += r;
        p |> r; sum += r;
        p |> r; sum += r;
        p |> r; sum += r;
        i += 1;
    }
            
    join t0;
    join t1;
    join t2;
    join t3;
    
    assert (sum == (((number_of_messages * 4) * 
                   ((number_of_messages * 4) - 1)) / 2));
}