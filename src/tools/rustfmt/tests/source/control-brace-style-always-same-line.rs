fn main() {
    loop {
        ();
        ();
        }


    'label: loop // loop comment  
    {
        ();
    }


    cond = true;
    while cond {
        ();
    }


    'while_label: while cond { // while comment
        ();
    }


    for obj in iter {
        for sub_obj in obj
        {
            'nested_while_label: while cond {
                ();
            }
        }
    }

    match some_var { // match comment
        pattern0 => val0,
        pattern1 => val1,
        pattern2 | pattern3 => {
            do_stuff();
            val2
        },
    };
}
