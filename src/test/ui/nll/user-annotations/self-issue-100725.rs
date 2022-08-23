// Make sure we suggest replacing `Self` with `Bigger` to make it pass.

// check-fail

pub struct Bigger<'a> {
    _marker: &'a (),
}
impl<'a> Bigger<'a> {
    pub fn get_addr(byte_list: &'a mut Vec<u8>) -> &mut u8 {
        byte_list.iter_mut().find_map(|item| {
            Self::other(item); // replace with `Bigger`
            Some(())
        });

        byte_list.push(0);
        //~^ ERROR cannot borrow `*byte_list` as mutable more than once at a time
        byte_list.last_mut().unwrap()
        //~^ ERROR cannot borrow `*byte_list` as mutable more than once at a time
    }

    pub fn other<'b: 'a>(_value: &'b mut u8) {
        todo!()
    }
}

fn main() {}
