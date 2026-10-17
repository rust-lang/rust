fn while_loop() {
    thread::spawn(|| {
        while i < days.len() {
            #![allow(clippy::indexing_slicing)]
            w |= days[i].bit_value();
            i += 1;
        }
    });
}

fn for_loop() {
    thread::spawn(|| {
        for i in 0..days.len() {
            #![allow(clippy::indexing_slicing)]
            w |= days[i].bit_value();
        }
    });
}

fn empty_loop_body() {
    thread::spawn(|| {
        while true {
            #![allow(clippy::indexing_slicing)]
        }
    });
}

fn closure_body_without_block() {
    thread::spawn(|| {
        while i < days.len() {
            #![allow(clippy::indexing_slicing)]
            w |= days[i].bit_value();
        }
    });
}

fn outer_attribute_still_formatted() {
    thread::spawn(|| {
        #[allow(clippy::indexing_slicing)]
        while i < days.len() {
            w |= days[i].bit_value();
        }
    });
}
