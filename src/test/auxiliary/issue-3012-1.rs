#[link(name="socketlib", vers="0.0")];

mod socket {

export socket_handle;

class socket_handle {
    let sockfd: libc::c_int;
    new(x: libc::c_int) {self.sockfd = x;}
    drop { /* c::close(self.sockfd); */ }
}
}
