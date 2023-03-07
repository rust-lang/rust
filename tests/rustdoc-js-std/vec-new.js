const QUERY = 'Vec::new';

const EXPECTED = {
    'others': [
        { 'path': 'std::vec::Vec', 'name': 'new' },
        { 'path': 'alloc::vec::Vec', 'name': 'new' },
        { 'path': 'std::vec::Vec', 'name': 'new_in' },
        { 'path': 'alloc::vec::Vec', 'name': 'new_in' },
    ],
};
