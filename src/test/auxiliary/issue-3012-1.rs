#[link(name="socketlib", vers="0.0")];
#[crate_type = "lib"];

mod socket {

export socket_handle;

struct socket_handle {
    let sockfd: libc::c_int;
    drop { /* c::close(self.sockfd); */ }
}

    fn socket_handle(x: libc::c_int) -> socket_handle {
        socket_handle {
            sockfd: x
        }
    }

}
