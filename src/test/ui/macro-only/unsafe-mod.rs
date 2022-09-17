// check-fail

unsafe mod m { //~ERROR module cannot be declared unsafe
    pub unsafe mod inner; //~ERROR module cannot be declared unsafe
                          //~|ERROR file not found for module
}

fn main() {}
