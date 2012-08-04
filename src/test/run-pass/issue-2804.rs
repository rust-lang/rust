use std;
import io;
import io::writer_util;
import std::map::hashmap;

enum object
{
    bool_value(bool),
    int_value(i64),
}

fn lookup(table: std::map::hashmap<~str, std::json::json>, key: ~str, default: ~str) -> ~str
{
    alt table.find(key)
    {
        option::some(std::json::string(s)) =>
        {
            *s
        }
        option::some(value) =>
        {
            error!{"%s was expected to be a string but is a %?", key, value};
            default
        }
        option::none =>
        {
            default
        }
    }
}

fn add_interface(store: int, managed_ip: ~str, data: std::json::json) -> (~str, object)
{
    alt data
    {
        std::json::dict(interface) =>
        {
            let name = lookup(interface, ~"ifDescr", ~"");
            let label = fmt!{"%s-%s", managed_ip, name};

            (label, bool_value(false))
        }
        _ =>
        {
            error!{"Expected dict for %s interfaces but found %?", managed_ip, data};
            (~"gnos:missing-interface", bool_value(true))
        }
    }
}

fn add_interfaces(store: int, managed_ip: ~str, device: std::map::hashmap<~str, std::json::json>) -> ~[(~str, object)]
{
    alt device[~"interfaces"]
    {
        std::json::list(interfaces) =>
        {
          do vec::map(*interfaces) |interface| {
                add_interface(store, managed_ip, interface)
          }
        }
        _ =>
        {
            error!{"Expected list for %s interfaces but found %?", managed_ip, device[~"interfaces"]};
            ~[]
        }
    }
}

fn main() {}
