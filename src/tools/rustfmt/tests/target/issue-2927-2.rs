// rustfmt-edition: 2015
#![feature(uniform_paths)]
use futures::prelude::*;
use http_03::cli::Cli;
use hyper::{service::service_fn_ok, Body, Response, Server};
use log::{error, info, log};
use structopt::StructOpt;
