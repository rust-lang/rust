fn main() { test00(); }

fn start() { #debug("Started / Finished task."); }

fn test00() {
    task::try(|| start() );
    #debug("Completing.");
}
