use std::io;
use std::fmt;
use std::io::{File};
use clean;

enum TagKind {
    Macro,
    Enumerator,
    Enumeration,
    Member,
    Function,
    Module,
    Structure,
    Variable,
    Typedef,
    Trait,
}

impl fmt::Show for TagKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let c = match *self {
            Macro => 'd',
            Enumerator => 'g',
            Enumeration => 'e',
            Member => 'm',
            Function => 'f',
            Module => 'n',
            Structure => 's',
            Variable => 'v',
            Typedef => 't',
            Trait => 'i'
        };

        write!( f, "{}", c )
    }
}

#[deriving(Clone)]
enum ExtraField {
    StructName(String),
    TraitName(String),
}

impl fmt::Show for ExtraField {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            StructName(ref n) => write!( f, "struct:{}", n ),
            TraitName(ref n) => write!( f, "trait:{}", n )
        }
    }
}

struct ExtraFields {
    fields: Vec<ExtraField>
}

impl ExtraFields {
    fn new( s: &[ExtraField] ) -> ExtraFields {
        ExtraFields { fields: Vec::from_slice( s ) }
    }
}

impl fmt::Show for ExtraFields {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, s) in self.fields.iter().enumerate() {
            if i != 0 {
                try!( write!( f, "\t" ) );
            }
            try!( s.fmt( f ) );
        }
        Ok(())
    }
}


struct Tag {
    symbol: String,
    file: String,
    location: String,
    kind: TagKind,
    extra: ExtraFields
}

impl Tag {
    fn from_item( item: &clean::Item, kind: TagKind, extra: &[ExtraField] ) -> Tag {
        let location = match item.source.tag_match {
            Some(ref x) => x.clone(),
            None => item.source.loline.to_string()
        };
        Tag {
            symbol: item.name.clone().expect( "unnamed item" ),
            file: item.source.filename.clone(),
            location: location,
            kind: kind,
            extra: ExtraFields::new(extra)
        }
    }
}

impl PartialOrd for Tag {
    fn partial_cmp( &self, other: &Tag ) -> Option<Ordering> {
        match self.symbol.cmp( &other.symbol ) {
            Equal => {},
            x => return Some(x)
        }

        match self.file.cmp( &other.file ) {
            Equal => {},
            x => return Some(x)
        }


        Some(self.location.cmp( &other.location ))
    }
}

impl Ord for Tag {
    fn cmp( &self, other: &Tag ) -> Ordering {
        self.partial_cmp( other ).unwrap()
    }
}


impl PartialEq for Tag {
    fn eq( &self, other: &Tag ) -> bool {
        self.symbol.eq( &other.symbol ) &&
            self.file.eq( &other.file ) &&
            self.location.eq( &other.location )
    }
}

impl Eq for Tag {}

macro_rules! inner(
    ($e:expr, $ty:ident) => (
        match $e.inner {
            clean::$ty(ref x) => x,
            _ => fail!( "expected `{}`", stringify!( $ty ) )
        }
    )
)

fn handle_struct_item( item: &clean::Item, tags: &mut Vec<Tag> ) {
    let _struct = inner!( item, StructItem );
    let struct_name = item.name.clone().expect("unnamed struct");

    tags.push( Tag::from_item( item, Structure, [] ) );
    
    for item in _struct.fields.iter() {
        if item.name.is_some() {
            tags.push( Tag::from_item(
                    item,
                    Member,
                    &[ StructName(struct_name.clone()) ]
                )
            );
        }
    }
}

fn handle_enum_item( item: &clean::Item, tags: &mut Vec<Tag> ) {
    let _enum = inner!( item, EnumItem );

    tags.push( Tag::from_item( item, Enumeration, [] ) );
    
    for item in _enum.variants.iter() {
        if item.name.is_some() {
            tags.push( Tag::from_item(
                    item,
                    Enumerator,
                    []
                )
            );
        }
    }
}

fn handle_trait_item( item: &clean::Item, tags: &mut Vec<Tag> ) {
    let _trait = inner!( item, TraitItem );
    let trait_name = item.name.clone().expect("unnamed trait");

    tags.push( Tag::from_item( item, Trait, [] ) );
    
    for ti in _trait.items.iter() {
        let item = ti.item();
        tags.push( Tag::from_item(
                item,
                Member,
                &[ TraitName(trait_name.clone()) ]
            )
        );
    }
}

fn handle_impl_item( item: &clean::Item, tags: &mut Vec<Tag> ) {
    let _impl = inner!( item, ImplItem );

    for item in _impl.items.iter() {
        tags.push( Tag::from_item(
                item,
                Function,
                []
            )
        );
    }
}



fn handle_module_item( item: &clean::Item, tags: &mut Vec<Tag> ) {
    let module = inner!( item, ModuleItem );

    for item in module.items.iter() {
        match item.inner {
            clean::StructItem(..) => handle_struct_item( item, tags ),
            clean::EnumItem(..) => handle_enum_item( item, tags ),
            clean::ModuleItem(..) => handle_module_item( item, tags ),
            clean::FunctionItem(..) => {
                tags.push( Tag::from_item( item, Function, [] ) );
            },
            clean::TypedefItem(..) => {
                tags.push( Tag::from_item( item, Typedef, [] ) );
            }
            clean::StaticItem(..) => {
                tags.push( Tag::from_item( item, Variable, [] ) );
            },
            clean::TraitItem(..) => handle_trait_item( item, tags ),
            clean::ImplItem(..) => handle_impl_item( item, tags ),
            clean::ViewItemItem(..) => {}, /* TODO - include crate/use aliases? */
            clean::TyMethodItem(..) => fail!( "ty method at module level" ),
            clean::MethodItem(..) => fail!( "method at module level" ),
            clean::StructFieldItem(..) => fail!( "struct field at module level" ),
            clean::VariantItem(..) => fail!( "variant at modile level" ),
            clean::ForeignFunctionItem(..) => {
                tags.push( Tag::from_item( item, Function, [] ) );
            },
            clean::ForeignStaticItem(..) => {
                tags.push( Tag::from_item( item, Variable, [] ) );
            },
            clean::MacroItem(..) => {
                tags.push( Tag::from_item( item, Macro, [] ) );
            },
            clean::PrimitiveItem(..) => fail!( "primitive" ),
        }
    }
}




pub fn output(krate: clean::Crate, dst: Path) -> io::IoResult<()> {
    let mut file = try!(File::create(&dst));

    let mut tags = Vec::new();

    let module_item = match krate.module {
        Some(m) => m,
        _ => return Ok(())
    };

    handle_module_item( &module_item, &mut tags );

    tags.sort();
    try!(writeln!(file, "!_TAG_FILE_FORMAT\t2"));
    try!(writeln!(file, "!_TAG_FILE_SORTED\t1"));
    try!(writeln!(file, "!_TAG_PROGRAM_NAME\trustdoc"));
    try!(writeln!(file, "!_TAG_PROGRAM_URL\thttp://www.rust-lang.org/"));
    try!(writeln!(file, "!_TAG_PROGRAM_VERSION\t{}", env!("CFG_VERSION")));

    for tag in tags.iter() {
        try!(
            writeln!( file, "{}\t{}\t/^{}$/;\" {}\t{}",
                  tag.symbol,
                  tag.file,
                  tag.location,
                  tag.kind,
                  tag.extra
            )
        );
    }

    Ok(())
}
