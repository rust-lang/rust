fn process_data(_: &[&[u8]]) {}

fn test_buffer_cleared_after_use() {
    let sources = vec![vec![1u8, 2, 3, 4, 5], vec![6, 7, 8, 9]];
    let mut buffer: Vec<&[u8]> = vec![];
    //~^ NOTE variable `buffer` declared here

    for source in sources {
        let data: Vec<u8> = source;
        //~^ NOTE binding `data` declared here
        buffer.extend(data.split(|x| *x == 3));
        //~^ ERROR `data` does not live long enough
        //~| NOTE borrowed value does not live long enough
        //~| NOTE borrow later used here
        //~| NOTE `buffer` is a collection that stores borrowed references, but `data` does not live long enough to be stored in it
        //~| HELP buffer reuse with borrowed references requires unsafe code or restructuring
        process_data(&buffer);
        buffer.clear();
    } //~ NOTE `data` dropped here while still borrowed
}

fn test_buffer_cleared_at_start() {
    let sources = vec![vec![1u8, 2, 3, 4, 5], vec![6, 7, 8, 9]];
    let mut buffer: Vec<&[u8]> = vec![];
    //~^ NOTE variable `buffer` declared here

    for source in sources {
        buffer.clear();
        //~^ NOTE borrow later used here
        let data: Vec<u8> = source;
        //~^ NOTE binding `data` declared here
        buffer.extend(data.split(|x| *x == 3));
        //~^ ERROR `data` does not live long enough
        //~| NOTE borrowed value does not live long enough
        //~| NOTE `buffer` is a collection that stores borrowed references, but `data` does not live long enough to be stored in it
        //~| HELP buffer reuse with borrowed references requires unsafe code or restructuring
        process_data(&buffer);
    } //~ NOTE `data` dropped here while still borrowed
}

fn test_no_explicit_clear() {
    let sources = vec![vec![1u8, 2, 3, 4, 5], vec![6, 7, 8, 9]];
    let mut buffer: Vec<&[u8]> = vec![];
    //~^ NOTE variable `buffer` declared here

    for source in sources {
        let data: Vec<u8> = source;
        //~^ NOTE binding `data` declared here
        buffer.extend(data.split(|x| *x == 3));
        //~^ ERROR `data` does not live long enough
        //~| NOTE borrowed value does not live long enough
        //~| NOTE borrow later used here
        //~| NOTE `buffer` is a collection that stores borrowed references, but `data` does not live long enough to be stored in it
        //~| HELP buffer reuse with borrowed references requires unsafe code or restructuring
        process_data(&buffer);
    } //~ NOTE `data` dropped here while still borrowed
}

fn main() {
    test_buffer_cleared_after_use();
    test_buffer_cleared_at_start();
    test_no_explicit_clear();
}
