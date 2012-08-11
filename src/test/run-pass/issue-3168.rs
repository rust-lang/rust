// xfail-fast

fn main() {
    let (c,p) = pipes::stream();
    do task::try {
        let (c2,p2) = pipes::stream();
        do task::spawn {
            p2.recv();
            #error["brother fails"];
            fail;
        }   
        let (c3,p3) = pipes::stream();
        c.send(c3);
        c2.send(());
        #error["child blocks"];
        p3.recv();
    };  
    #error["parent tries"];
    assert !p.recv().try_send(());
    #error("all done!");
}
