#![crate_name="socketlib"]
#![crate_type = "lib"]

pub mod socket {
    pub struct socket_handle {
        sockfd: u32,
    }

    impl Drop for socket_handle {
        fn drop(&mut self) {
            /* c::close(self.sockfd); */
        }
    }

    pub fn socket_handle(x: u32) -> socket_handle {
        socket_handle {
            sockfd: x
        }
    }
}
