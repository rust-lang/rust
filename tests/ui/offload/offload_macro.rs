#![feature(gpu_offload)]

fn kernel() {}

fn main() {
    core::offload::offload! { args = () }
    //~^ ERROR missing `kernel`

    core::offload::offload! { kernel = kernel }
    //~^ ERROR missing `args`

    core::offload::offload! { kernel = kernel, args = (), foo = 1 }
    //~^ ERROR unknown field `foo`

    core::offload::offload! { kernel = kernel, kernel = kernel, args = () }
    //~^ ERROR duplicate field `kernel`

    core::offload::offload! { kernel = kernel, args = (), args = () }
    //~^ ERROR duplicate field `args`

    core::offload::offload! { workgroup_dim = [1, 1, 1], workgroup_dim = [1, 1, 1] }
    //~^ ERROR duplicate field `workgroup_dim`

    core::offload::offload! { thread_dim = [32, 1, 1], thread_dim = [64, 1, 1] }
    //~^ ERROR duplicate field `thread_dim`

    core::offload::offload! { kernel = kernel, args = (), dyn_cache = 0, dyn_cache = 8 }
    //~^ ERROR duplicate field `dyn_cache`

    core::offload::offload! { kernel = kernel, args = (), device = 0, device = 1 }
    //~^ ERROR duplicate field `device`
}
