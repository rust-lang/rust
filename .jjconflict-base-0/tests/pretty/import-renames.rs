//@ compile-flags: --crate-type=lib

//@ pp-exact

use std::io::{self, Error as IoError};
use std::net::{self as stdnet, TcpStream};
