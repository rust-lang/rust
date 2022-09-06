#!/usr/bin/env python

# This test ensures that every ID in the produced json actually resolves to an item either in
# `index` or `paths`. It DOES NOT check that the structure of the produced json is actually in
# any way correct, for example an empty map would pass.

# FIXME: Better error output

import sys
import json

crate = json.load(open(sys.argv[1], encoding="utf-8"))


def get_local_item(item_id):
    if item_id in crate["index"]:
        return crate["index"][item_id]
    print("Missing local ID:", item_id)
    sys.exit(1)


# local IDs have to be in `index`, external ones can sometimes be in `index` but otherwise have
# to be in `paths`
def valid_id(item_id):
    return item_id in crate["index"] or item_id[0] != "0" and item_id in crate["paths"]


def check_generics(generics):
    for param in generics["params"]:
        check_generic_param(param)
    for where_predicate in generics["where_predicates"]:
        if "bound_predicate" in where_predicate:
            pred = where_predicate["bound_predicate"]
            check_type(pred["type"])
            for bound in pred["bounds"]:
                check_generic_bound(bound)
        elif "region_predicate" in where_predicate:
            pred = where_predicate["region_predicate"]
            for bound in pred["bounds"]:
                check_generic_bound(bound)
        elif "eq_predicate" in where_predicate:
            pred = where_predicate["eq_predicate"]
            check_type(pred["rhs"])
            check_type(pred["lhs"])


def check_generic_param(param):
    if "type" in param["kind"]:
        ty = param["kind"]["type"]
        if ty["default"]:
            check_type(ty["default"])
    elif "const" in param["kind"]:
        check_type(param["kind"]["const"])


def check_generic_bound(bound):
    if "trait_bound" in bound:
        for param in bound["trait_bound"]["generic_params"]:
            check_generic_param(param)
        check_path(bound["trait_bound"]["trait"])


def check_decl(decl):
    for (_name, ty) in decl["inputs"]:
        check_type(ty)
    if decl["output"]:
        check_type(decl["output"])

def check_path(path):
    args = path["args"]
    if args:
        if "angle_bracketed" in args:
            for arg in args["angle_bracketed"]["args"]:
                if "type" in arg:
                    check_type(arg["type"])
                elif "const" in arg:
                    check_type(arg["const"]["type"])
            for binding in args["angle_bracketed"]["bindings"]:
                if "equality" in binding["binding"]:
                    term = binding["binding"]["equality"]
                    if "type" in term: check_type(term["type"])
                    elif "const" in term: check_type(term["const"])
                elif "constraint" in binding["binding"]:
                    for bound in binding["binding"]["constraint"]:
                        check_generic_bound(bound)
        elif "parenthesized" in args:
            for input_ty in args["parenthesized"]["inputs"]:
                check_type(input_ty)
            if args["parenthesized"]["output"]:
                check_type(args["parenthesized"]["output"])
    if not valid_id(path["id"]):
        print("Type contained an invalid ID:", path["id"])
        sys.exit(1)

def check_type(ty):
    if ty["kind"] == "resolved_path":
        check_path(ty["inner"])
    elif ty["kind"] == "tuple":
        for ty in ty["inner"]:
            check_type(ty)
    elif ty["kind"] == "slice":
        check_type(ty["inner"])
    elif ty["kind"] == "impl_trait":
        for bound in ty["inner"]:
            check_generic_bound(bound)
    elif ty["kind"] in ("raw_pointer", "borrowed_ref", "array"):
        check_type(ty["inner"]["type"])
    elif ty["kind"] == "function_pointer":
        for param in ty["inner"]["generic_params"]:
            check_generic_param(param)
        check_decl(ty["inner"]["decl"])
    elif ty["kind"] == "qualified_path":
        check_type(ty["inner"]["self_type"])
        check_path(ty["inner"]["trait"])


work_list = set([crate["root"]])
visited = work_list.copy()

while work_list:
    current = work_list.pop()
    visited.add(current)
    item = get_local_item(current)
    # check intradoc links
    for (_name, link) in item["links"].items():
        if not valid_id(link):
            print("Intra-doc link contains invalid ID:", link)

    # check all fields that reference types such as generics as well as nested items
    # (modules, structs, traits, and enums)
    if item["kind"] == "module":
        work_list |= set(item["inner"]["items"]) - visited
    elif item["kind"] == "struct":
        check_generics(item["inner"]["generics"])
        work_list |= (
            set(item["inner"]["fields"]) | set(item["inner"]["impls"])
        ) - visited
    elif item["kind"] == "struct_field":
        check_type(item["inner"])
    elif item["kind"] == "enum":
        check_generics(item["inner"]["generics"])
        work_list |= (
            set(item["inner"]["variants"]) | set(item["inner"]["impls"])
        ) - visited
    elif item["kind"] == "variant":
        if item["inner"]["variant_kind"] == "tuple":
            for field_id in filter(None, item["inner"]["variant_inner"]):
                work_list.add(field_id)
        elif item["inner"]["variant_kind"] == "struct":
            work_list |= set(item["inner"]["variant_inner"]["fields"]) - visited
    elif item["kind"] in ("function", "method"):
        check_generics(item["inner"]["generics"])
        check_decl(item["inner"]["decl"])
    elif item["kind"] in ("static", "constant", "assoc_const"):
        check_type(item["inner"]["type"])
    elif item["kind"] == "typedef":
        check_type(item["inner"]["type"])
        check_generics(item["inner"]["generics"])
    elif item["kind"] == "opaque_ty":
        check_generics(item["inner"]["generics"])
        for bound in item["inner"]["bounds"]:
            check_generic_bound(bound)
    elif item["kind"] == "trait_alias":
        check_generics(item["inner"]["params"])
        for bound in item["inner"]["bounds"]:
            check_generic_bound(bound)
    elif item["kind"] == "trait":
        check_generics(item["inner"]["generics"])
        for bound in item["inner"]["bounds"]:
            check_generic_bound(bound)
        work_list |= (
            set(item["inner"]["items"]) | set(item["inner"]["implementations"])
        ) - visited
    elif item["kind"] == "impl":
        check_generics(item["inner"]["generics"])
        if item["inner"]["trait"]:
            check_path(item["inner"]["trait"])
        if item["inner"]["blanket_impl"]:
            check_type(item["inner"]["blanket_impl"])
        check_type(item["inner"]["for"])
        for assoc_item in item["inner"]["items"]:
            if not valid_id(assoc_item):
                print("Impl block referenced a missing ID:", assoc_item)
                sys.exit(1)
    elif item["kind"] == "assoc_type":
        for bound in item["inner"]["bounds"]:
            check_generic_bound(bound)
        if item["inner"]["default"]:
            check_type(item["inner"]["default"])
    elif item["kind"] == "import":
        if item["inner"]["id"]:
            inner_id = item["inner"]["id"]
            assert valid_id(inner_id)
            if inner_id in crate["index"] and inner_id not in visited:
                work_list.add(inner_id)
